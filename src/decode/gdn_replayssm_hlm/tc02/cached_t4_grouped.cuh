// SPDX-License-Identifier: Apache-2.0
// TC02 cached T4, with one warp for each 16 V rows and FP32 substitution.
template <bool OperandA, class Fragment>
__device__ __forceinline__ void t4_load_qk(Fragment& fragment, const BF* base, int lane) {
  const BF* pair = base + (lane/4)*128 + 2*(lane%4);
  uint32_t words[4]{};
  words[0] = *reinterpret_cast<const uint32_t*>(pair);
  words[OperandA ? 2 : 1] = *reinterpret_cast<const uint32_t*>(pair+8);
  static_assert(sizeof(fragment.x) == sizeof(words));
  __builtin_memcpy(fragment.x,words,sizeof(words));
}

template <bool CarryHistory, class Fragment>
__device__ __forceinline__ void t4_load_keys(Fragment& fragment, const BF* normalized,
                                            const BF* old_k, int k, int lane) {
  const int r = (lane&7) + (lane&16)/2, c = lane&8;
  const BF* row = normalized + (r%4)*2*128;
  if constexpr (CarryHistory)
    if (r < 12) row = old_k + r*128;
  const uint4 words = cached_tsm::load4<false>(row+k+c);
  static_assert(sizeof(fragment.x) == sizeof(words));
  __builtin_memcpy(fragment.x,&words,sizeof(words));
}

template <int Warps, class Params>
__device__ __forceinline__ void t4_normalize_grouped(const Params& p, BF* qk, int row,
                                                    int head, int warp, int lane) {
  constexpr int K=128,T=4,Group=8*Warps,Items=K/Group;
  const int t=warp*(T/Warps)+lane/Group;
  const int col=(lane%Group)*Items;
  const int64_t token=int64_t(row)*T+t;
  const BF* q=static_cast<const BF*>(p.q.data)+token*p.q.stride[0]+head*K+col;
  const BF* k=static_cast<const BF*>(p.k.data)+token*p.k.stride[0]+head*K+col;
  float a[Items],b[Items];
  if constexpr (Items==4) {
    load_bf16x4(q,a);
    load_bf16x4(k,b);
  } else {
    #pragma unroll
    for (int i=0; i<Items; i+=8) {
      float next_a[8],next_b[8];
      load_bf16x8(q+i,next_a);
      load_bf16x8(k+i,next_b);
      #pragma unroll
      for (int j=0; j<8; ++j) {
        a[i+j]=next_a[j];
        b[i+j]=next_b[j];
      }
    }
  }
  float q2=0.f,k2=0.f;
  #pragma unroll
  for (int i=0; i<Items; ++i) {
    q2+=a[i]*a[i];
    k2+=b[i]*b[i];
  }
  #pragma unroll
  for (int mask=Group/2; mask>0; mask/=2) {
    q2+=__shfl_xor_sync(0xffffffff,q2,mask,Group);
    k2+=__shfl_xor_sync(0xffffffff,k2,mask,Group);
  }
  const float qs=p.scale/(sqrtf(q2)+p.epsilon),ks=1.f/(sqrtf(k2)+p.epsilon);
  #pragma unroll
  for (int i=0; i<Items; i+=2) {
    const uint32_t a0=__bfloat16_as_ushort(BF(a[i]*qs)),a1=__bfloat16_as_ushort(BF(a[i+1]*qs));
    const uint32_t b0=__bfloat16_as_ushort(BF(b[i]*ks)),b1=__bfloat16_as_ushort(BF(b[i+1]*ks));
    *reinterpret_cast<uint32_t*>(qk+(2*t+1)*K+col+i)=a0|(a1<<16);
    *reinterpret_cast<uint32_t*>(qk+(2*t)*K+col+i)=b0|(b1<<16);
  }
}

template <class A, class B>
__device__ __forceinline__ void t4_load_qk_pair(A& a, B& b, const BF* base, int lane) {
  const BF* pair=base+(lane/4)*128+2*(lane%4);
  const uint32_t x=*reinterpret_cast<const uint32_t*>(pair),y=*reinterpret_cast<const uint32_t*>(pair+8);
  const uint32_t words_a[4]={x,0,y,0},words_b[4]={x,y,0,0};
  __builtin_memcpy(a.x,words_a,sizeof(words_a));
  __builtin_memcpy(b.x,words_b,sizeof(words_b));
}

template <int TileV, int Heads, bool Single, bool FuseKeys, bool VecNorm, bool ShareQk, class Params>
__global__ __launch_bounds__(2*TileV) void cached_t4_tensor(Params p) {
  constexpr int T=4, K=128, V=128, L=16, WarpV=16, Warps=TileV/WarpV;
  constexpr int HistoryWarp = Warps==1 ? 0 : 1;
  constexpr int GateWarp = Warps>2 ? 2 : HistoryWarp;
  static_assert(TileV==16 || TileV==32 || TileV==64);
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  const int row=Single ? 0 : blockIdx.y, hv=blockIdx.z, head=Heads==1 ? 0 : hv/4;
  const int vbase=blockIdx.x*TileV, local_v=warp*WarpV;
  const int64_t slot = static_cast<const int32_t*>(p.slots.data)[row];
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && slot < p.replay_slots);
  BF* const state = static_cast<BF*>(p.state.data) + slot*p.state.stride[0] + int64_t(hv*V+vbase)*K;

  __shared__ __align__(128) BF old_k[L*K], old_u[L*TileV], low_u[L*TileV], state_stage[TileV*K];
  __shared__ __align__(32) BF qk[2*T*K];
  __shared__ float history_g[L], total[T], prefix[T], beta[T], origin;

  const bool copies_state = warp == 0;
  if (copies_state)
    cached_bulk_bf16(state_stage,state,K,TileV,K,TileV,0,K);
  const int count = static_cast<const int32_t*>(p.lengths.data)[slot];
  const int bank = static_cast<const int32_t*>(p.buffers.data)[slot];
  assert(0 <= count && count <= L && 0 <= bank && bank < 2);
  const bool has_history=count!=0, overflow=count+T>L;
  const int64_t kb=slot*p.keys.stride[0] + bank*p.keys.stride[1] + head*K;
  const int64_t ub=slot*p.values.stride[0] + bank*p.values.stride[1] + hv*V;
  const int64_t gb=slot*p.gates.stride[0] + bank*p.gates.stride[1] + hv*L;
  const bool copies_history=has_history && warp==HistoryWarp;
  if (copies_history) {
    cached_bulk_bf16(old_k,static_cast<const BF*>(p.keys.data)+kb,K,count,K,L,0,p.keys.stride[2]);
    cached_bulk_bf16(old_u,static_cast<const BF*>(p.values.data)+ub,V,count,TileV,L,vbase,p.values.stride[2]);
  }
  if (copies_state || copies_history) __pipeline_commit();

  if (warp==GateWarp) {
    // The same warp provides the history gate vector and its final entry.
    const float h = lane < count ? read<float>(p.gates,gb+lane) : 0.f;
    const float start = __shfl_sync(0xffffffff,h,has_history ? count-1 : 0);
    const int64_t token = int64_t(row)*T+lane;
    const float g = lane < T ? read<float>(p.a,token*p.a.stride[0]+hv) : 0.f;
    const float b = lane < T ? read<float>(p.b,token*p.b.stride[0]+hv) : 0.f;
    float scan = 0.f;
    #pragma unroll
    for (int i=0; i<T; ++i) {
      const float next = __shfl_sync(0xffffffff,g,i);
      if (i <= lane) scan += next;
    }
    if (lane < L) history_g[lane] = h;
    if (lane < T) {
      prefix[lane] = scan;
      total[lane] = start+scan;
      beta[lane] = b;
    }
    if (lane==0) origin = start;
  }

  if constexpr (VecNorm) {
    t4_normalize_grouped<Warps>(p,qk,row,head,warp,lane);
  } else {
    #pragma unroll
    for (int t=warp; t<T; t+=Warps) {
    const int64_t token=int64_t(row)*T+t;
    float raw_q[4],raw_k[4],q2=0.f,k2=0.f;
    #pragma unroll
    for (int i=0; i<4; ++i) {
      const int c=lane+32*i;
      raw_q[i] = read<BF>(p.q,token*p.q.stride[0]+head*K+c);
      raw_k[i] = read<BF>(p.k,token*p.k.stride[0]+head*K+c);
      q2 += raw_q[i]*raw_q[i];
      k2 += raw_k[i]*raw_k[i];
    }
    const float qs = p.scale/(sqrtf(sum_warp(q2))+p.epsilon);
    const float ks = 1.f/(sqrtf(sum_warp(k2))+p.epsilon);
    #pragma unroll
    for (int i=0; i<4; ++i) {
      const int c=lane+32*i;
      qk[(2*t)*K+c]=BF(raw_k[i]*ks);
      qk[(2*t+1)*K+c]=BF(raw_q[i]*qs);
    }
    }
  }
  const int t = lane%T, v = vbase+local_v+lane/T;
  const int64_t token=int64_t(row)*T+t;
  const BF* const input=static_cast<const BF*>(p.v.data)+token*p.v.stride[0]+hv*V+v;
  const BF value0=input[0],value1=input[8];
  if (copies_state || copies_history) __pipeline_wait_prior(0);
  if constexpr (Warps==1) __syncwarp();
  else __syncthreads();

  using RowA=awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major>;
  using ColA=awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major>;
  using ColB=awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major>;
  using RowB=awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
  using Acc=awmma::fragment<awmma::accumulator,16,16,16,float>;
  Acc projected,history,new_corr,old_corr;
  awmma::fill_fragment(projected,0.f);
  awmma::fill_fragment(history,0.f);
  awmma::fill_fragment(new_corr,0.f);
  awmma::fill_fragment(old_corr,0.f);
  const BF* const staged_state=state_stage+local_v*K;
  // Ordinary rounds can put the four current keys in unused history columns.
  const bool mixed=FuseKeys && has_history && !overflow;
  #pragma unroll 2
  for (int k=0; k<K; k+=16) {
    RowA query, s;
    ColB key, current;
    if constexpr (ShareQk) t4_load_qk_pair(query,current,qk+k,lane);
    else t4_load_qk<true>(query,qk+k,lane);
    if (mixed)
      t4_load_keys<true>(key,qk,old_k,k,lane);
    else
      t4_load_keys<false>(key,qk,old_k,k,lane);
    awmma::mma_sync(new_corr,query,key,new_corr);
    if (has_history && !mixed) {
      load_b16(key,old_k+k,K,lane);
      awmma::mma_sync(old_corr,query,key,old_corr);
    }
    load_a16(s,staged_state+k,K,lane);
    if constexpr (!ShareQk) t4_load_qk<false>(current,qk+k,lane);
    awmma::mma_sync(projected,s,current,projected);
  }
  const float gt=total[t], bt=beta[t], d=expf(gt);
  if (has_history) {
    ColA u;
    ColB coefficient;
    uint32_t words[4]{};
    const float total_row=total[lane/8];
    #pragma unroll
    for (int i=0; i<2; ++i) {
      const int j=2*(lane%4)+8*i;
      const float c0=mixed ? new_corr.x[4*i] : old_corr.x[4*i];
      const float c1=mixed ? new_corr.x[4*i+1] : old_corr.x[4*i+1];
      const BF first=j<count ? BF(c0*expf(total_row-history_g[j])) : BF(0.f);
      const BF second=j+1<count ? BF(c1*expf(total_row-history_g[j+1])) : BF(0.f);
      words[i]=uint32_t(__bfloat16_as_ushort(first))|(uint32_t(__bfloat16_as_ushort(second))<<16);
    }
    __builtin_memcpy(coefficient.x,words,sizeof(words));
    load_fold_operand<true>(u,old_u+local_v,TileV,lane);
    awmma::mma_sync(history,u,coefficient,history);
  }

  // Each subgroup owns one V row, with one time step per lane.
  float u0=bt*(float(value0)-d*projected.x[0]-history.x[0]);
  float u1=bt*(float(value1)-d*projected.x[2]-history.x[2]);
  float out0=d*projected.x[1]+history.x[1],out1=d*projected.x[3]+history.x[3];
  #pragma unroll
  for (int j=0; j<T; ++j) {
    const float datum = mixed ? new_corr.x[4+j%2] : new_corr.x[j%2];
    const int col_group = (mixed ? 2 : 0) + j/2;
    const float kk=__shfl_sync(0xffffffff,datum,8*t+col_group);
    const float qk_coef=__shfl_sync(0xffffffff,datum,8*t+4+col_group);
    const float factor=expf(gt-total[j]);
    const float previous0=__shfl_sync(0xffffffff,u0,j,T);
    const float previous1=__shfl_sync(0xffffffff,u1,j,T);
    if (t > j) {
      const float a=bt*kk*factor;
      u0-=a*previous0;
      u1-=a*previous1;
    }
    if (t >= j) {
      const float b=qk_coef*factor;
      out0+=b*previous0;
      out1+=b*previous1;
    }
  }
  const int dest=overflow ? 1-bank : bank, position=overflow ? 0 : count;
  const int64_t output=token*p.out.stride[0]+hv*V+v;
  const int64_t value=slot*p.values.stride[0]+dest*p.values.stride[1]+(position+t)*p.values.stride[2]+hv*V+v;
  write<BF>(p.out,output,out0);
  write<BF>(p.out,output+8,out1);
  write<BF>(p.values,value,u0);
  write<BF>(p.values,value+8,u1);
  if (blockIdx.x==0) {
    if ((hv&3)==0)
      for (int i=tid; i<T*K; i+=2*TileV)
        static_cast<BF*>(p.keys.data)[slot*p.keys.stride[0]+dest*p.keys.stride[1]+
            (position+i/K)*p.keys.stride[2]+head*K+i%K]=qk[(i/K)*2*K+i%K];
    if (tid<T)
      write<float>(p.gates,slot*p.gates.stride[0]+dest*p.gates.stride[1]+hv*L+position+tid,
                   overflow ? prefix[tid] : total[tid]);
  }

  if (p.materialize_overflow && overflow) {
    __syncwarp();
    #pragma unroll
    for (int i=lane; i<L*WarpV; i+=32) {
      const int j=i/WarpV, offset=j*TileV+local_v+i%WarpV;
      const float scaled=j<count ? float(old_u[offset])*expf(origin-history_g[j]) : 0.f;
      const BF hi=BF(scaled);
      old_u[offset]=hi;
      low_u[offset]=BF(scaled-float(hi));
    }
    __syncwarp();
    ColA hi_u,lo_u;
    load_fold_operand<true>(hi_u,old_u+local_v,TileV,lane);
    load_fold_operand<true>(lo_u,low_u+local_v,TileV,lane);
    const float state_decay=expf(origin);
    #pragma unroll 1
    for (int k=0; k<K; k+=16) {
      RowB key;
      Acc hi,lo;
      load_fold_operand<false>(key,old_k+k,K,lane);
      awmma::fill_fragment(hi,0.f);
      awmma::fill_fragment(lo,0.f);
      awmma::mma_sync(hi,hi_u,key,hi);
      awmma::mma_sync(lo,lo_u,key,lo);
      #pragma unroll
      for (int i=0; i<4; ++i) {
        const int r=lane/4+(i%2)*8, c=k+(lane%4)*2+(i/2)*8;
        const uint32_t old=*reinterpret_cast<const uint32_t*>(staged_state+r*K+c);
        const float s0=__uint_as_float(old<<16),s1=__uint_as_float(old&0xffff0000u);
        const uint32_t x0=__bfloat16_as_ushort(BF(s0*state_decay+(hi.x[2*i]+lo.x[2*i])));
        const uint32_t x1=__bfloat16_as_ushort(BF(s1*state_decay+(hi.x[2*i+1]+lo.x[2*i+1])));
        *reinterpret_cast<uint32_t*>(state+(local_v+r)*K+c)=x0|(x1<<16);
      }
    }
  }
}
