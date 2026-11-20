// SPDX-License-Identifier: Apache-2.0
template <int GROUP>
__device__ __forceinline__ float cached_group_sum(float value) {
  static_assert(GROUP == 2 || GROUP == 4 || GROUP == 8, "GROUP must be 2, 4, or 8");
  #pragma unroll
  for (int offset = GROUP / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

// T=1 launch: grid.x partitions the value dimension; each CTA uses 128 threads.
template <class S, int GROUP = 4, bool MMA_FOLD = false>
__global__ __launch_bounds__(128) void cached_decode_vector_kernel(CachedVerify p) {
  static_assert(GROUP == 2 || GROUP == 4 || GROUP == 8, "GROUP must be 2, 4, or 8");
  constexpr int K = 128, BV = 128 / GROUP, CAP = 16, PIECES = 128 / (GROUP * 4);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.y, hv = blockIdx.z, head = hv / (p.value_heads / p.heads);
  const int vbase = blockIdx.x * BV, v = vbase + tid / GROUP, klane = tid & (GROUP-1);
  const int64_t slot = index(p.slots,row);
  if (slot <= p.null_slot) return;
  const int64_t replay = index(p.replay,row);
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t cr = index(p.lengths,replay), br = index(p.buffers,replay);
  assert(cr >= 0 && cr <= CAP && br >= 0 && br < 2);
  if (p.offsets.data) assert(index(p.offsets,replay) == 0);
  const int count = int(cr), bank = int(br), dest = count == CAP ? 1-bank : bank, wp = count == CAP ? 0 : count;
  const int64_t kb = replay*p.keys.stride[0]+bank*p.keys.stride[1]+head*K;
  const int64_t ub = replay*p.values.stride[0]+bank*p.values.stride[1]+hv*128;
  const int64_t gb = replay*p.gates.stride[0]+bank*p.gates.stride[1]+hv*CAP;
  __shared__ __align__(32) float q[K], k[K], keys[CAP*K], old_u[CAP*BV], gh[CAP], ck[CAP], cq[CAP];
  __shared__ __align__(32) float upper[CAP*BV], lower[CAP*BV], scalars[6];
  for (int i = tid; i < count*K; i += 128)
    keys[i] = read<BF>(p.keys,kb+(i/K)*p.keys.stride[2]+i%K);
  for (int i = tid; i < count*BV; i += 128)
    old_u[i] = read<BF>(p.values,ub+(i/BV)*p.values.stride[2]+vbase+i%BV);
  if (tid < count) gh[tid] = read<float>(p.gates,gb+tid);
  if (warp == 0) {
    float sq = 0.f, sk = 0.f;
    #pragma unroll
    for (int c = lane; c < K; c += 32) {
      const float a = read<BF>(p.q,int64_t(row)*p.q.stride[0]+head*K+c);
      const float b = read<BF>(p.k,int64_t(row)*p.k.stride[0]+head*K+c);
      sq += a*a; sk += b*b;
    }
    sq = sum_warp(sq); sk = sum_warp(sk);
    const float qs = p.scale/norm_denominator(sq,p.norm_policy,p.epsilon);
    const float ks = 1.f/norm_denominator(sk,p.norm_policy,p.epsilon);
    #pragma unroll
    for (int c = lane; c < K; c += 32) {
      q[c] = rounded<BF>(read<BF>(p.q,int64_t(row)*p.q.stride[0]+head*K+c)*qs);
      k[c] = rounded<BF>(read<BF>(p.k,int64_t(row)*p.k.stride[0]+head*K+c)*ks);
    }
  }
  if (tid == 0) {
    float g = read_any(p.a,int64_t(row)*p.a.stride[0]+hv);
    float b = read_any(p.b,int64_t(row)*p.b.stride[0]+hv);
    if (!p.processed_gates) {
      const float arg = g+read<float>(p.dt_bias,hv*p.dt_bias.stride[0]);
      g = -expf(read<float>(p.A_log,hv*p.A_log.stride[0])) *
          (arg <= 20.f ? 0.6931471805599453f*__log2f(__fadd_rn(1.f,__expf(arg))) : arg);
      b = 1.f/(1.f+expf(-b));
    }
    const float origin = count ? read<float>(p.gates,gb+count-1) : 0.f;
    scalars[0] = origin; scalars[1] = origin+g; scalars[2] = expf(origin+g);
    scalars[3] = b; scalars[4] = g;
  }
  __syncthreads();
  for (int j = warp; j < count+1; j += 4) {
    float a = 0.f, b = 0.f;
    #pragma unroll
    for (int c = lane; c < K; c += 32) {
      const float key = j < count ? keys[j*K+c] : k[c];
      a += k[c]*key; b += q[c]*key;
    }
    a = sum_warp(a); b = sum_warp(b);
    if (lane == 0) {
      if (j < count) {
        const float d = expf(scalars[1]-gh[j]);
        ck[j] = rounded<BF>(a*d); cq[j] = rounded<BF>(b*d);
      } else scalars[5] = b;
    }
  }
  float state[128 / GROUP];
  float kp = 0.f, qp = 0.f, kpl = 0.f, qpl = 0.f;
  #pragma unroll
  for (int i = 0; i < PIECES; ++i) {
    float values[4];
    const int c = klane*4+i*(GROUP*4);
    read_vector<S,4>(p.state,slot*p.state.stride[0]+(int64_t(hv)*128+v)*K+c,4,values);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      if constexpr (!MMA_FOLD) state[i*4+e] = values[e];
      const float h = rounded<BF>(values[e]);
      kp += h*k[c+e]; qp += h*q[c+e];
      if constexpr (std::is_same_v<S,float>) {
        const float lo = rounded<BF>(values[e]-h);
        kpl += lo*k[c+e]; qpl += lo*q[c+e];
      }
    }
  }
  kp = cached_group_sum<GROUP>(kp)+cached_group_sum<GROUP>(kpl); qp = cached_group_sum<GROUP>(qp)+cached_group_sum<GROUP>(qpl);
  __syncthreads();
  float kh = 0.f, qh = 0.f;
  for (int j = klane; j < count; j += GROUP) {
    kh += ck[j]*old_u[j*BV+tid/GROUP]; qh += cq[j]*old_u[j*BV+tid/GROUP];
  }
  kh = cached_group_sum<GROUP>(kh); qh = cached_group_sum<GROUP>(qh);
  if (klane == 0) {
    const float u = scalars[3]*(read<BF>(p.v,int64_t(row)*p.v.stride[0]+hv*128+v)-scalars[2]*kp-kh);
    write<BF>(p.out,int64_t(row)*p.out.stride[0]+hv*128+v,scalars[2]*qp+qh+scalars[5]*u);
    write<BF>(p.values,replay*p.values.stride[0]+dest*p.values.stride[1]+wp*p.values.stride[2]+hv*128+v,u);
  }
  if (blockIdx.x == 0) {
    if (hv % (p.value_heads/p.heads) == 0)
      write<BF>(p.keys,replay*p.keys.stride[0]+dest*p.keys.stride[1]+wp*p.keys.stride[2]+head*K+tid,k[tid]);
    if (tid == 0) write<float>(p.gates,replay*p.gates.stride[0]+dest*p.gates.stride[1]+hv*CAP+wp,count == CAP ? scalars[4] : scalars[1]);
  }
  if (p.materialize_overflow && count == CAP) {
    if constexpr (MMA_FOLD) {
      static_assert(BV == 16 || BV == 32, "MMA Fold requires GROUP=4 or GROUP=8");
      static_assert(sizeof(BF) == 2 && sizeof(float) == 4, "Fold storage requires BF16 and FP32");
      static_assert(4*256*sizeof(float)+CAP*K*sizeof(BF) <= CAP*K*sizeof(float),
                    "Fold key staging and warp scratch must fit within keys");
      // Finish all candidate/history reads before repurposing shared storage.
      __syncthreads();
      BF* upperBF = reinterpret_cast<BF*>(upper);
      BF* lowerBF = reinterpret_cast<BF*>(lower);
      BF* keyBF = reinterpret_cast<BF*>(keys);
      for (int i = tid; i < CAP*BV; i += 128) {
        const float value = old_u[i]*expf(scalars[0]-gh[i/BV]);
        const BF hi = BF(value);
        upperBF[i] = hi; lowerBF[i] = BF(value-float(hi));
      }
      // Reread the untouched GLOBAL old bank, never convert keys in place.
      for (int i = tid; i < CAP*K; i += 128)
        keyBF[i] = static_cast<const BF*>(p.keys.data)[kb+(i/K)*p.keys.stride[2]+i%K];
      __syncthreads();
      using AF = awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major>;
      using BFg = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
      using CF = awmma::fragment<awmma::accumulator,16,16,16,float>;
      static_assert(CF::num_elements == 8, "Fold expects eight accumulator elements per lane");
      constexpr int tiles = (BV/16)*(K/16);
      float* scratch = keys+1024+warp*256;
      const float d = expf(scalars[0]);
      for (int tile = warp; tile < tiles; tile += 4) {
        const int vt = (tile/(K/16))*16, kt = (tile%(K/16))*16;
        AF a; BFg b; CF hi,lo,sum;
        awmma::fill_fragment(hi,0.f); awmma::fill_fragment(lo,0.f);
        awmma::load_matrix_sync(a,upperBF+vt,BV);
        awmma::load_matrix_sync(b,keyBF+kt,K);
        awmma::mma_sync(hi,a,b,hi);
        awmma::load_matrix_sync(a,lowerBF+vt,BV);
        awmma::mma_sync(lo,a,b,lo);
        #pragma unroll
        for (int e = 0; e < 8; ++e) sum.x[e] = hi.x[e]+lo.x[e];
        awmma::store_matrix_sync(scratch,sum,16,awmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
          const int64_t off = slot*p.state.stride[0]+(int64_t(hv)*128+vbase+vt+i/16)*K+kt+i%16;
          const float old = read<S>(p.state,off);
          write<S>(p.state,off,old*d+scratch[i]);
        }
        __syncwarp();
      }
    } else {
      for (int i = tid; i < CAP*BV; i += 128) {
        const float value = old_u[i]*expf(scalars[0]-gh[i/BV]);
        upper[i] = rounded<BF>(value); lower[i] = rounded<BF>(value-upper[i]);
      }
      __syncthreads();
      const float d = expf(scalars[0]);
      #pragma unroll
      for (int i = 0; i < PIECES; ++i) {
        const int c = klane*4+i*(GROUP*4);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          float delta = 0.f;
          #pragma unroll
          for (int j = 0; j < CAP; ++j) {
            delta += keys[j*K+c+e]*upper[j*BV+tid/GROUP];
            delta += keys[j*K+c+e]*lower[j*BV+tid/GROUP];
          }
          write<S>(p.state,slot*p.state.stride[0]+(int64_t(hv)*128+v)*K+c+e,state[i*4+e]*d+delta);
        }
      }
    }
  }
}

template <class View>
__device__ __forceinline__ void cached_copy_bf8(const View& t, int64_t offset, int valid, BF* dst) {
  if (valid == 0) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) dst[e] = BF(0.f);
    return;
  }
  const BF* src = static_cast<const BF*>(t.data) + offset;
  if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst) = raw;
  } else {
    #pragma unroll
    for (int e = 0; e < 8; ++e) dst[e] = src[e];
  }
}

template <class S, int N, class View>
__device__ __forceinline__ void cached_fold_store_vector(const View& t, int64_t offset, const float (&values)[N]) {
  static_assert((std::is_same_v<S,float> && N == 4) || (std::is_same_v<S,BF> && N == 8),
                "Fold vector stores require FP32x4 or BF16x8");
  S* dst = static_cast<S*>(t.data) + offset;
  if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
    if constexpr (std::is_same_v<S,float>) {
      *reinterpret_cast<float4*>(dst) = float4{values[0],values[1],values[2],values[3]};
    } else {
      uint32_t words[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t lo = static_cast<uint32_t>(__bfloat16_as_ushort(BF(values[2*i])));
        const uint32_t hi = static_cast<uint32_t>(__bfloat16_as_ushort(BF(values[2*i+1])));
        words[i] = lo | (hi << 16);
      }
      *reinterpret_cast<uint4*>(dst) = uint4{words[0],words[1],words[2],words[3]};
    }
  } else {
    #pragma unroll
    for (int e = 0; e < N; ++e) write<S>(t,offset+e,values[e]);
  }
}

// State staging and projection outputs share storage only across synchronized phases.
template <class S, int T, int BV, int LEVEL = 1, bool INLINE_FOLD = false>
__global__ __launch_bounds__(128) void cached_tensor_kernel(CachedVerify p) {
  constexpr int R = 16, K = 128, CAP = 16;
  constexpr bool SPLIT = std::is_same_v<S,float>;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.y, hv = blockIdx.z, head = hv / (p.value_heads / p.heads);
  const int64_t slot = index(p.slots, row);
  if (slot <= p.null_slot) return;
  const int64_t replay = index(p.replay, row);
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t cr = index(p.lengths, replay), br = index(p.buffers, replay);
  assert(cr >= 0 && cr <= CAP && br >= 0 && br < 2);
  if (p.offsets.data) assert(index(p.offsets, replay) == 0);
  const int count = int(cr), bank = int(br), vbase = blockIdx.x * BV;
  const bool overflow = count + T > CAP;
  const int write_bank = overflow ? 1 - bank : bank, wp = overflow ? 0 : count;
  const int64_t kb = replay * p.keys.stride[0] + bank * p.keys.stride[1] + head * K;
  const int64_t ub = replay * p.values.stride[0] + bank * p.values.stride[1] + hv * 128;
  const int64_t gb = replay * p.gates.stride[0] + bank * p.gates.stride[1] + hv * CAP;
  __shared__ __align__(32) BF x[R * K], old_k[CAP * K];
  __shared__ __align__(32) BF coeff[R * CAP], old_u[CAP * BV];
  __shared__ __align__(32) float proj[R * BV], proj_lo[SPLIT ? R * BV : 1], history[R * BV];
  BF* state_hi = reinterpret_cast<BF*>(proj);
  BF* state_lo = reinterpret_cast<BF*>(proj_lo);
  __shared__ __align__(32) float corr[R * 32];
  __shared__ float g[T], total[T], beta[T], gh[CAP], origin, decay[T];
  for (int i = tid; i < R * K; i += 128) x[i] = BF(0.f);
  #pragma unroll
  for (int i = tid * 8; i < CAP * K; i += 128 * 8) {
    cached_copy_bf8(p.keys,kb+(i/K)*p.keys.stride[2]+i%K,i/K<count ? 8 : 0,old_k+i);
  }
  #pragma unroll
  for (int i = tid * 8; i < CAP * BV; i += 128 * 8) {
    cached_copy_bf8(p.values,ub+(i/BV)*p.values.stride[2]+vbase+i%BV,i/BV<count ? 8 : 0,old_u+i);
  }
  if (tid < CAP) gh[tid] = tid < count ? read<float>(p.gates, gb + tid) : 0.f;
  if (tid == 0) origin = count ? read<float>(p.gates, gb + count - 1) : 0.f;
  __syncthreads();
  for (int t = warp; t < T; t += 4) {
    float sq = 0.f, sk = 0.f, raw_q[4], raw_k[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = lane+i*32;
      raw_q[i] = read<BF>(p.q, int64_t(row * T + t) * p.q.stride[0] + head * K + c);
      raw_k[i] = read<BF>(p.k, int64_t(row * T + t) * p.k.stride[0] + head * K + c);
      sq += raw_q[i]*raw_q[i]; sk += raw_k[i]*raw_k[i];
    }
    sq = sum_warp(sq); sk = sum_warp(sk);
    const float qs = p.scale / norm_denominator(sq, p.norm_policy, p.epsilon);
    const float ks = 1.f / norm_denominator(sk, p.norm_policy, p.epsilon);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = lane+i*32;
      x[t * K + c] = BF(raw_k[i]*ks);
      x[(T + t) * K + c] = BF(raw_q[i]*qs);
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
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    #pragma unroll
    for (int t = 0; t < T; ++t) { acc += g[t]; total[t] = origin + acc; }
  }
  __syncthreads();
  if constexpr (LEVEL < 2) {
    for (int pair = warp; pair < 2 * T * (T + count); pair += 4) {
      const int r = pair / (T + count), j = pair % (T + count);
      const BF* key = j < T ? x + j * K : old_k + (j - T) * K;
      float val = 0.f;
      #pragma unroll
      for (int c = lane; c < K; c += 32) val += float(x[r * K + c]) * float(key[c]);
      val = sum_warp(val);
      if (lane == 0) corr[r * 32 + j] = val;
    }
  } else {
    awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major> a;
    awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major> b;
    awmma::fragment<awmma::accumulator,16,16,16,float> c;
    if (warp < 2) {
      awmma::fill_fragment(c,0.f);
      #pragma unroll
      for (int kk = 0; kk < K; kk += 16) {
        awmma::load_matrix_sync(a,x + kk,K);
        awmma::load_matrix_sync(b,(warp == 0 ? x : old_k) + kk,K);
        awmma::mma_sync(c,a,b,c);
      }
      awmma::store_matrix_sync(corr + warp * 16,c,32,awmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int i = tid; i < R * CAP; i += 128) {
    const int r = i / CAP, j = i % CAP, t = r % T;
    const int col = LEVEL < 2 ? T + j : 16 + j;
    coeff[i] = BF(r < 2 * T && j < count ? corr[r * 32 + col] * expf(total[t] - gh[j]) : 0.f);
  }
  using AF = awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major>;
  using BFg = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major>;
  using CF = awmma::fragment<awmma::accumulator,16,16,16,float>;
  constexpr int PER_WARP = (BV + 63) / 64;
  CF high[PER_WARP], low[PER_WARP];
  #pragma unroll
  for (int i = 0; i < PER_WARP; ++i) {
    awmma::fill_fragment(high[i],0.f);
    if constexpr (SPLIT) awmma::fill_fragment(low[i],0.f);
  }
  constexpr int LOAD = sizeof(S) == 2 ? 8 : 4;
  bool async_staging = false;
  if constexpr (std::is_same_v<S,BF> && !SPLIT) {
    const BF* state = static_cast<const BF*>(p.state.data) + slot * p.state.stride[0] +
        (int64_t(hv) * 128 + vbase) * K;
    if ((reinterpret_cast<uintptr_t>(state) & 15) == 0) {
      async_staging = true;
      BF* state_alt = reinterpret_cast<BF*>(history);
      auto load_stage = [&](int phase, BF* dst) {
        #pragma unroll
        for (int i = tid * 8; i < BV * 32; i += 128 * 8) {
          __pipeline_memcpy_async(dst + i, static_cast<const BF*>(p.state.data) +
              slot * p.state.stride[0] + (int64_t(hv) * 128 + vbase + i / 32) * K +
              phase * 32 + i % 32,16);
        }
        __pipeline_commit();
      };
      load_stage(0,state_hi);
      auto consume_stage = [&](auto tag) {
        constexpr int Phase = decltype(tag)::value;
        BF* current = Phase % 2 == 0 ? state_hi : state_alt;
        if constexpr (Phase < 3) {
          load_stage(Phase + 1,Phase % 2 == 0 ? state_alt : state_hi);
          __pipeline_wait_prior(1);
        } else {
          __pipeline_wait_prior(0);
        }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < PER_WARP; ++s) {
          const int vt = warp + s * 4;
          if (vt < BV / 16) {
            AF a; BFg b;
            #pragma unroll
            for (int part = 0; part < 32; part += 16) {
              awmma::load_matrix_sync(a,x + Phase * 32 + part,K);
              awmma::load_matrix_sync(b,current + vt * 512 + part,32);
              awmma::mma_sync(high[s],a,b,high[s]);
            }
          }
        }
        __syncthreads();
      };
      consume_stage(std::integral_constant<int,0>{});
      consume_stage(std::integral_constant<int,1>{});
      consume_stage(std::integral_constant<int,2>{});
      consume_stage(std::integral_constant<int,3>{});
    }
  }
  if (!async_staging) {
  for (int kt = 0; kt < K; kt += 32) {
    #pragma unroll
    for (int i = tid * LOAD; i < BV * 32; i += 128 * LOAD) {
      const int vi = i / 32, ki = i % 32;
      if constexpr (std::is_same_v<S,BF> && !SPLIT) {
        cached_copy_bf8(p.state, slot * p.state.stride[0] + (int64_t(hv) * 128 + vbase + vi) * K + kt + ki,8,state_hi+i);
        continue;
      }
      float values[LOAD];
      read_vector<S,LOAD>(p.state, slot * p.state.stride[0] + (int64_t(hv) * 128 + vbase + vi) * K + kt + ki,LOAD,values);
      #pragma unroll
      for (int e = 0; e < LOAD; ++e) {
        const BF h = BF(values[e]);
        state_hi[i+e] = h;
        if constexpr (SPLIT) state_lo[i+e] = BF(values[e]-float(h));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < PER_WARP; ++s) {
      const int vt = warp + s * 4;
      if (vt < BV / 16) {
        AF a; BFg b;
        #pragma unroll
        for (int part = 0; part < 32; part += 16) {
          awmma::load_matrix_sync(a,x + kt + part,K);
          awmma::load_matrix_sync(b,state_hi + vt * 512 + part,32);
          awmma::mma_sync(high[s],a,b,high[s]);
          if constexpr (SPLIT) {
            awmma::load_matrix_sync(b,state_lo + vt * 512 + part,32);
            awmma::mma_sync(low[s],a,b,low[s]);
          }
        }
      }
    }
    __syncthreads();
  }
  }
  #pragma unroll
  for (int s = 0; s < PER_WARP; ++s) {
    const int vt = warp + s * 4;
    if (vt < BV / 16) {
      awmma::store_matrix_sync(proj + vt * 16,high[s],BV,awmma::mem_row_major);
      if constexpr (SPLIT)
        awmma::store_matrix_sync(proj_lo + vt * 16,low[s],BV,awmma::mem_row_major);
    }
  }
  __syncthreads();
  if constexpr (LEVEL >= 3) {
    #pragma unroll
    for (int vt = warp; vt < BV / 16; vt += 4) {
      AF a;
      awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major> b;
      CF c;
      awmma::fill_fragment(c,0.f);
      awmma::load_matrix_sync(a,coeff,CAP);
      awmma::load_matrix_sync(b,old_u + vt * 16,BV);
      awmma::mma_sync(c,a,b,c);
      awmma::store_matrix_sync(history + vt * 16,c,BV,awmma::mem_row_major);
    }
  } else {
    for (int i = tid; i < 2 * T * BV; i += 128) {
      const int r = i / BV, v = i % BV;
      float acc = 0.f;
      for (int j = 0; j < count; ++j) acc += float(coeff[r * CAP + j]) * float(old_u[j * BV + v]);
      history[i] = acc;
    }
  }
  __syncthreads();
  if (tid < T) decay[tid] = expf(total[tid]);
  for (int i = tid; i < 2 * T * T; i += 128) {
    const int r = i / T, j = i % T, t = r % T;
    const float value = corr[r * 32 + j];
    corr[r * 32 + j] = (r < T ? beta[t] * value : value) * expf(total[t]-total[j]);
  }
  __syncthreads();
  for (int v = tid; v < BV; v += 128) {
    float u[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      const float d = decay[t];
      float update = beta[t] * (read<BF>(p.v,int64_t(row * T + t) * p.v.stride[0] + hv * 128 + vbase + v) -
          d * (proj[t * BV + v] + (SPLIT ? proj_lo[t * BV + v] : 0.f)) - history[t * BV + v]);
      #pragma unroll
      for (int j = 0; j < t; ++j) update -= corr[t * 32 + j] * u[j];
      u[t] = update;
      float output = d * (proj[(T+t) * BV + v] + (SPLIT ? proj_lo[(T+t) * BV + v] : 0.f)) + history[(T+t) * BV + v];
      #pragma unroll
      for (int j = 0; j <= t; ++j) output += corr[(T+t) * 32 + j] * u[j];
      write<BF>(p.out,int64_t(row * T + t) * p.out.stride[0] + hv * 128 + vbase + v,output);
      write<BF>(p.values,replay * p.values.stride[0] + write_bank * p.values.stride[1] + (wp+t) * p.values.stride[2] + hv * 128 + vbase + v,update);
    }
  }
  if (blockIdx.x == 0) {
    if (hv % (p.value_heads / p.heads) == 0) {
      for (int i = tid; i < T * K; i += 128)
        static_cast<BF*>(p.keys.data)[replay * p.keys.stride[0] + write_bank * p.keys.stride[1] + (wp+i/K) * p.keys.stride[2] + head * K + i%K] = x[i];
    }
    if (tid == 0) {
      float acc = 0.f;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        acc += g[t];
        write<float>(p.gates,replay * p.gates.stride[0] + write_bank * p.gates.stride[1] + hv * CAP + wp+t,overflow ? acc : total[t]);
      }
    }
  }
  if constexpr (INLINE_FOLD) {
    if (p.materialize_overflow && overflow) {
      __syncthreads();
      for (int i = tid; i < CAP * BV; i += 128) {
        const int j = i / BV;
        const float scaled = j < count ? float(old_u[i]) * expf(origin-gh[j]) : 0.f;
        const BF high = BF(scaled);
        old_u[i] = high; x[i] = BF(scaled-float(high));
      }
      __syncthreads();
      static_assert(CF::num_elements == 8, "Fold expects eight accumulator elements per lane");
      const float state_decay = expf(origin);
      for (int vt = warp; vt < BV / 16; vt += 4) {
        awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major> a_hi,a_lo;
        awmma::load_matrix_sync(a_hi,old_u+vt*16,BV);
        awmma::load_matrix_sync(a_lo,x+vt*16,BV);
        for (int kt = 0; kt < K; kt += 32) {
          #pragma unroll
          for (int part = 0; part < 2; ++part) {
            awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major> b;
            CF hi,lo,sum;
            awmma::fill_fragment(hi,0.f); awmma::fill_fragment(lo,0.f);
            awmma::load_matrix_sync(b,old_k+kt+part*16,K);
            awmma::mma_sync(hi,a_hi,b,hi);
            awmma::mma_sync(lo,a_lo,b,lo);
            #pragma unroll
            for (int e = 0; e < 8; ++e) sum.x[e] = hi.x[e]+lo.x[e];
            awmma::store_matrix_sync((part == 0 ? proj : history)+vt*256,sum,16,awmma::mem_row_major);
          }
          __syncwarp();
          for (int i = lane*LOAD; i < 512; i += 32*LOAD) {
            const int row = i / 32, col = i % 32;
            const int64_t off = slot*p.state.stride[0]+(int64_t(hv)*128+vbase+vt*16+row)*K+kt+col;
            float values[LOAD];
            read_vector<S,LOAD>(p.state,off,LOAD,values);
            const float* scratch = col < 16 ? proj : history;
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

// Called after the runtime PPU1.5 check and the fixed-layout routes.
bool launch_cached_tensor(const CachedVerify& p, hggcStream_t stream) {
  if (p.q.dtype == BF16 && p.key_dim == 128 && p.value_dim == 128 &&
      p.capacity == 16 && !p.starts.data && p.tf32 && p.norm_policy == 2) {
    if (p.max_steps == 1) {
      if (p.state.dtype == F32) {
        if (int64_t(p.batch)*p.value_heads < 128)
          cached_decode_vector_kernel<float,8><<<dim3(8,p.batch,p.value_heads),128,0,stream>>>(p);
        else
          cached_decode_vector_kernel<float,4><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p);
      } else {
        if (int64_t(p.batch)*p.value_heads >= 128)
          cached_tensor_kernel<BF,1,128,3,true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p);
        else
          cached_decode_vector_kernel<BF,4,false><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p);
      }
      return true;
    }
#define CACHED_STEP(N) case N: \
    if (N <= 4 && int64_t(p.batch)*p.value_heads >= 128) { \
      if (p.state.dtype == F32) \
        cached_tensor_kernel<float, N, 64, 3, true><<<dim3(2,p.batch,p.value_heads),128,0,stream>>>(p); \
      else if (int64_t(p.batch)*p.value_heads >= 512) \
        cached_tensor_kernel<BF, N, 128, 3, true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p); \
      else \
        cached_tensor_kernel<BF, N, 64, 3, true><<<dim3(2,p.batch,p.value_heads),128,0,stream>>>(p); \
    } else if (N <= 4 || int64_t(p.batch)*p.value_heads < 128) { \
      if (p.state.dtype == F32) \
        cached_tensor_kernel<float, N, 32, 3, true><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p); \
      else \
        cached_tensor_kernel<BF, N, 32, 3, true><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p); \
    } else { \
      if (p.state.dtype == F32) \
        cached_tensor_kernel<float, N, 128, 3, true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p); \
      else \
        cached_tensor_kernel<BF, N, 128, 3, true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p); \
    } \
    return true
    switch (p.max_steps) {
      CACHED_STEP(1); CACHED_STEP(2); CACHED_STEP(3); CACHED_STEP(4);
      CACHED_STEP(5); CACHED_STEP(6); CACHED_STEP(7); CACHED_STEP(8);
    }
#undef CACHED_STEP
  }
  return false;
}
