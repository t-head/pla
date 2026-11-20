// SPDX-License-Identifier: Apache-2.0
// PPU1.5 fixed-length cached update. One warp owns 16 V rows and keeps
// its projections and forward substitution in TensorCell fragments.
template <bool OperandA, int T, class Fragment>
__device__ __forceinline__ void register_load_qk(Fragment& out, const BF* base, int lane) {
  if constexpr (T==1) {
    cached_two_rows<OperandA>(out,base,128,lane);
  } else if constexpr (T==8) {
    if constexpr (OperandA) load_a16(out,base,128,lane);
    else load_b16(out,base,128,lane);
  } else {
    const BF* pair=base+(lane/4)*128+2*(lane%4);
    uint32_t words[4]{};
    words[0]=*reinterpret_cast<const uint32_t*>(pair);
    words[OperandA ? 2 : 1]=*reinterpret_cast<const uint32_t*>(pair+8);
    __builtin_memcpy(out.x,words,sizeof(words));
  }
}

template <int T, bool History, int CurrentColumns=T, class Fragment>
__device__ __forceinline__ void register_load_keys(Fragment& out, const BF* qk,
                                                  const BF* old_k, int k, int lane) {
  const int row=(lane&7)+(lane&16)/2, col=lane&8;
  const BF* source=qk+(row%T)*256;
  if constexpr (History) {
    constexpr int HistoryColumns=16-CurrentColumns;
    source=row<HistoryColumns ? old_k+row*128 : qk+(row-HistoryColumns)*256;
  }
  const uint4 words=cached_tsm::load4<false>(source+k+col);
  static_assert(sizeof(out.x)==sizeof(words));
  __builtin_memcpy(out.x,&words,sizeof(words));
}

template <class A, class B>
__device__ __forceinline__ void register_load_t4_pair(A& a,B& b,const BF* qk,int lane) {
  const BF* pair=qk+(lane/4)*128+2*(lane%4);
  const uint32_t x=*reinterpret_cast<const uint32_t*>(pair),y=*reinterpret_cast<const uint32_t*>(pair+8);
  const uint32_t rows[4]={x,0,y,0},cols[4]={x,y,0,0};
  __builtin_memcpy(a.x,rows,sizeof(rows));
  __builtin_memcpy(b.x,cols,sizeof(cols));
}

__device__ __forceinline__ uint32_t register_pack(BF first, BF second) {
  return uint32_t(__bfloat16_as_ushort(first))|(uint32_t(__bfloat16_as_ushort(second))<<16);
}

template <class Fragment>
__device__ __forceinline__ void register_load_float_a(Fragment& hi, Fragment& lo,
                                                      const float* state, int lane) {
  uint32_t high[4],low[4];
  #pragma unroll
  for (int i=0; i<4; ++i) {
    const int r=lane/4+8*(i%2), c=2*(lane%4)+8*(i/2);
    const float2 x=*reinterpret_cast<const float2*>(state+r*128+c);
    const BF h0=BF(x.x),h1=BF(x.y);
    high[i]=register_pack(h0,h1);
    low[i]=register_pack(BF(x.x-float(h0)),BF(x.y-float(h1)));
  }
  __builtin_memcpy(hi.x,high,sizeof(high));
  __builtin_memcpy(lo.x,low,sizeof(low));
}

template <int T, int Warps, bool LastDot=false, class Params>
__device__ __forceinline__ float register_normalize(const Params& p, BF* qk, int row,
                                                    int head, int warp, int lane) {
  constexpr int Group=T==1 ? 32 : (32*Warps)/T, Items=128/Group;
  const int t=T==1 ? 0 : warp*(32/Group)+lane/Group, col=(lane%Group)*Items;
  const int64_t token=int64_t(row)*T+t;
  const BF* q=static_cast<const BF*>(p.q.data)+token*p.q.stride[0]+head*128+col;
  const BF* k=static_cast<const BF*>(p.k.data)+token*p.k.stride[0]+head*128+col;
  float a[Items],b[Items];
  if constexpr (Items==4) {
    load_bf16x4(q,a);
    load_bf16x4(k,b);
  } else {
    #pragma unroll
    for (int i=0; i<Items; i+=8) {
      float x[8],y[8];
      load_bf16x8(q+i,x);
      load_bf16x8(k+i,y);
      #pragma unroll
      for (int j=0; j<8; ++j) {
        a[i+j]=x[j];
        b[i+j]=y[j];
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
  float dot=0.f;
  #pragma unroll
  for (int i=0; i<Items; i+=2) {
    const BF q0=BF(a[i]*qs),q1=BF(a[i+1]*qs),k0=BF(b[i]*ks),k1=BF(b[i+1]*ks);
    *reinterpret_cast<uint32_t*>(qk+(2*t+1)*128+col+i)=register_pack(q0,q1);
    *reinterpret_cast<uint32_t*>(qk+2*t*128+col+i)=register_pack(k0,k1);
    if constexpr (T==1 || LastDot) dot+=float(q0)*float(k0)+float(q1)*float(k1);
  }
  if constexpr (T==1) return sum_warp(dot);
  if constexpr (LastDot)
    for (int mask=Group/2; mask>0; mask/=2)
      dot+=__shfl_xor_sync(0xffffffff,dot,mask,Group);
  if constexpr (LastDot) return dot;
  return 0.f;
}

template <class S, int T, int TileV, int Heads, bool Single, bool FuseKeys, int FloatLayout, int TinyOptions, int Unroll, class Params>
__global__ __launch_bounds__(2*TileV) void cached_register_tensor(Params p) {
  constexpr int K=128,V=128,L=16,WarpV=16,Warps=TileV/WarpV;
  constexpr int HistoryWarp=Warps==1 ? 0 : 1, GateWarp=Warps>2 ? 2 : HistoryWarp;
  constexpr bool Split=std::is_same_v<S,float>, StagedSplit=Split && FloatLayout==0;
  constexpr bool RawStage=Split && FloatLayout==2;
  constexpr int Subgroup=4, TimeBlocks=T==8 ? 2 : 1;
  constexpr bool ThreeKeys=(TinyOptions&1)!=0, ReadyInput=(TinyOptions&2)!=0, SharedQk=(TinyOptions&4)!=0;
  constexpr bool GateFactors=(TinyOptions&8)!=0;
  static_assert(T==1 || T==4 || T==8);
  static_assert(TileV==16 || TileV==32 || TileV==64);
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const int row=Single ? 0 : blockIdx.y,hv=blockIdx.z,head=hv/Params::head_group;
  const int vbase=blockIdx.x*TileV,local_v=warp*WarpV;
  const int64_t slot=static_cast<const int32_t*>(p.slots.data)[row];
  if (slot<=p.null_slot) return;
  assert(slot>=0 && slot<p.state_slots && slot<p.replay_slots);
  S* const state=static_cast<S*>(p.state.data)+slot*p.state.stride[0]+int64_t(hv*V+vbase)*K;

  __shared__ __align__(128) BF old_k[L*K],old_u[L*TileV],low_u[L*TileV];
  __shared__ __align__(128) BF state_hi[Split && !StagedSplit ? 1 : TileV*K];
  __shared__ __align__(128) float state_raw[RawStage ? TileV*K : 1];
  __shared__ __align__(128) BF state_lo[StagedSplit ? TileV*K : 1];
  __shared__ __align__(32) BF qk[2*T*K];
  __shared__ float history_g[L],total[T],prefix[T],beta[T],origin,qk_dot;
  __shared__ float weights[GateFactors ? T*T : 1];

  const bool copies_state=(!Split || RawStage) && warp==0;
  if constexpr (!Split)
    if (copies_state) cached_bulk_bf16(state_hi,state,K,TileV,K,TileV,0,K);
  if constexpr (RawStage)
    if (copies_state)
      cached_bulk_bf16(reinterpret_cast<BF*>(state_raw),reinterpret_cast<const BF*>(state),
                       2*K,TileV,2*K,TileV,0,2*K);
  const int count=static_cast<const int32_t*>(p.lengths.data)[slot];
  const int bank=static_cast<const int32_t*>(p.buffers.data)[slot];
  assert(0<=count && count<=L && 0<=bank && bank<2);
  const bool has_history=count!=0,overflow=count+T>L;
  const int64_t kb=slot*p.keys.stride[0]+bank*p.keys.stride[1]+head*K;
  const int64_t ub=slot*p.values.stride[0]+bank*p.values.stride[1]+hv*V;
  const int64_t gb=slot*p.gates.stride[0]+bank*p.gates.stride[1]+hv*L;
  const bool copies_history=has_history && warp==HistoryWarp;
  if (copies_history) {
    cached_bulk_bf16(old_k,static_cast<const BF*>(p.keys.data)+kb,K,count,K,L,0,p.keys.stride[2]);
    cached_bulk_bf16(old_u,static_cast<const BF*>(p.values.data)+ub,V,count,TileV,L,vbase,p.values.stride[2]);
  }
  if (copies_state || copies_history) __pipeline_commit();

  if constexpr (StagedSplit) {
    #pragma unroll 2
    for (int i=4*tid; i<TileV*K; i+=8*TileV) {
      const float4 x=*reinterpret_cast<const float4*>(state+i);
      const BF a=BF(x.x),b=BF(x.y),c=BF(x.z),d=BF(x.w);
      *reinterpret_cast<uint2*>(state_hi+i)=make_uint2(register_pack(a,b),register_pack(c,d));
      *reinterpret_cast<uint2*>(state_lo+i)=make_uint2(
          register_pack(BF(x.x-float(a)),BF(x.y-float(b))),
          register_pack(BF(x.z-float(c)),BF(x.w-float(d))));
    }
  }

  if (warp==GateWarp) {
    const float previous=lane<count ? read<float>(p.gates,gb+lane) : 0.f;
    const float start=__shfl_sync(0xffffffff,previous,has_history ? count-1 : 0);
    const int64_t token=int64_t(row)*T+lane;
    float g = 0.f, b = 0.f;
    if (lane < T) {
      if constexpr (Params::raw_gates) {
        const float x = read<BF>(p.a, token*p.a.stride[0]+hv) + read<float>(p.dt_bias, hv);
        const float softplus = x <= 20.f ?
            0.6931471805599453f * __log2f(__fadd_rn(1.f, __expf(x))) : x;
        g = -expf(read<float>(p.A_log, hv)) * softplus;
        b = 1.f / (1.f + expf(-read<BF>(p.b, token*p.b.stride[0]+hv)));
      } else {
        g = read<float>(p.a, token*p.a.stride[0]+hv);
        b = read<float>(p.b, token*p.b.stride[0]+hv);
      }
    }
    float scan=0.f;
    #pragma unroll
    for (int i=0; i<T; ++i) {
      const float next=__shfl_sync(0xffffffff,g,i);
      if (i<=lane) scan+=next;
    }
    if (lane<L) history_g[lane]=previous;
    if (lane<T) {
      prefix[lane]=scan;
      total[lane]=start+scan;
      beta[lane]=b;
    }
    if (lane==0) origin=start;
    if constexpr (GateFactors) {
      const float target=__shfl_sync(0xffffffff,start+scan,(lane/4)&3);
      const float source=__shfl_sync(0xffffffff,start+scan,lane&3);
      if (lane<16) weights[lane]=expf(target-source);
    }
  }
  if constexpr (T==1) {
    if (warp==0) {
      const float dot=register_normalize<T,Warps>(p,qk,row,head,warp,lane);
      if (lane==0) qk_dot=dot;
    }
  } else if constexpr (ThreeKeys) {
    constexpr int Group=32*Warps/T;
    const float dot=register_normalize<T,Warps,true>(p,qk,row,head,warp,lane);
    if (warp*(32/Group)+lane/Group==3 && lane%Group==0) qk_dot=dot;
  } else {
    register_normalize<T,Warps>(p,qk,row,head,warp,lane);
  }

  const int t=T==1 ? 0 : lane%Subgroup,v=vbase+local_v+lane/Subgroup;
  BF input0,input1;
  if constexpr (ReadyInput) {
    const int64_t token=int64_t(row)*T+t;
    const BF* input=static_cast<const BF*>(p.v.data)+token*p.v.stride[0]+hv*V+v;
    input0=input[0];
    input1=input[8];
  }
  if (copies_state || copies_history) __pipeline_wait_prior(0);
  if constexpr (Warps==1) __syncwarp();
  else __syncthreads();

  using RowA=awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major>;
  using ColA=awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major>;
  using ColB=awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major>;
  using RowB=awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
  using Acc=awmma::fragment<awmma::accumulator,16,16,16,float>;
  Acc projected,projected_lo,history,new_corr,old_corr;
  awmma::fill_fragment(projected,0.f);
  awmma::fill_fragment(history,0.f);
  awmma::fill_fragment(new_corr,0.f);
  awmma::fill_fragment(old_corr,0.f);
  if constexpr (Split) awmma::fill_fragment(projected_lo,0.f);
  const bool mixed=FuseKeys && has_history && (ThreeKeys ? count<=13 : !overflow);

  #pragma unroll Unroll
  for (int k=0; k<K; k+=16) {
    RowA query,s;
    ColB key,current;
    if constexpr (SharedQk) register_load_t4_pair(query,current,qk+k,lane);
    else if (T!=1 || has_history) register_load_qk<true,T>(query,qk+k,lane);
    if constexpr (T!=1) {
      if (mixed) register_load_keys<T,true,ThreeKeys ? 3 : T>(key,qk,old_k,k,lane);
      else register_load_keys<T,false>(key,qk,old_k,k,lane);
      awmma::mma_sync(new_corr,query,key,new_corr);
    }
    if (has_history && !mixed) {
      load_b16(key,old_k+k,K,lane);
      awmma::mma_sync(old_corr,query,key,old_corr);
    }
    RowA lo;
    if constexpr (Split && !StagedSplit) {
      register_load_float_a(s,lo,(RawStage ? state_raw : state)+local_v*K+k,lane);
    } else {
      load_a16(s,state_hi+local_v*K+k,K,lane);
      if constexpr (Split) load_a16(lo,state_lo+local_v*K+k,K,lane);
    }
    if constexpr (!SharedQk) register_load_qk<false,T>(current,qk+k,lane);
    awmma::mma_sync(projected,s,current,projected);
    if constexpr (Split) awmma::mma_sync(projected_lo,lo,current,projected_lo);
  }
  if constexpr (Split) {
    #pragma unroll
    for (int i=0; i<8; ++i) projected.x[i]+=projected_lo.x[i];
  }
  if (has_history) {
    ColA u;
    ColB coefficient;
    uint32_t words[4]{};
    if (T!=1 || lane<8) {
      #pragma unroll
      for (int th=0; th<TimeBlocks; ++th) {
        const float gt=total[lane/8+4*th];
        #pragma unroll
        for (int i=0; i<2; ++i) {
          const int j=2*(lane%4)+8*i,index=4*i+2*th;
          const float first_dot=mixed ? new_corr.x[index] : old_corr.x[index];
          const float second_dot=mixed ? new_corr.x[index+1] : old_corr.x[index+1];
          const BF first=j<count ? BF(first_dot*expf(gt-history_g[j])) : BF(0.f);
          const BF second=j+1<count ? BF(second_dot*expf(gt-history_g[j+1])) : BF(0.f);
          words[2*th+i]=register_pack(first,second);
        }
      }
    }
    __builtin_memcpy(coefficient.x,words,sizeof(words));
    load_fold_operand<true>(u,old_u+local_v,TileV,lane);
    awmma::mma_sync(history,u,coefficient,history);
  }

  const int dest=overflow ? 1-bank : bank,position=overflow ? 0 : count;
  // Four adjacent lanes own times t=0..3 of each V row.
  if constexpr (T==1) {
    if ((lane&3)==0) {
      const float decay=expf(total[0]);
      const int64_t token=row;
      #pragma unroll
      for (int i=0; i<2; ++i) {
        const int offset=hv*V+v+8*i;
        const float value=read<BF>(p.v,token*p.v.stride[0]+offset);
        const float u=beta[0]*(value-decay*projected.x[2*i]-history.x[2*i]);
        const float out=decay*projected.x[2*i+1]+history.x[2*i+1]+qk_dot*u;
        write<BF>(p.out,token*p.out.stride[0]+offset,out);
        write<BF>(p.values,slot*p.values.stride[0]+dest*p.values.stride[1]+position*p.values.stride[2]+offset,u);
      }
    }
  } else {
    float u[TimeBlocks][2],out[TimeBlocks][2],gt[TimeBlocks],bt[TimeBlocks];
    #pragma unroll
    for (int th=0; th<TimeBlocks; ++th) {
      const int64_t token=int64_t(row)*T+t+4*th;
      gt[th]=total[t+4*th];
      bt[th]=beta[t+4*th];
      const float d=expf(gt[th]);
      #pragma unroll
      for (int i=0; i<2; ++i) {
        const int index=4*th+2*i,vo=hv*V+v+8*i;
        const float value=ReadyInput ? float(i==0 ? input0 : input1)
            : read<BF>(p.v,token*p.v.stride[0]+vo);
        u[th][i]=bt[th]*(value-d*projected.x[index]-history.x[index]);
        out[th][i]=d*projected.x[index+1]+history.x[index+1];
      }
    }
    #pragma unroll
    for (int j=0; j<T; ++j) {
      const float previous0=__shfl_sync(0xffffffff,u[j/4][0],j%4,Subgroup);
      const float previous1=__shfl_sync(0xffffffff,u[j/4][1],j%4,Subgroup);
      constexpr int FusedStart=ThreeKeys ? 13 : 16-T;
      const int column=j+(mixed ? FusedStart : 0);
      const int source=ThreeKeys ? 8*t+(column%8)/2 : 8*t+j/2+(T==4 && mixed ? 2 : 0);
      const float total_j=total[j];
      #pragma unroll
      for (int th=j/4; th<TimeBlocks; ++th) {
        const int index=ThreeKeys ? column%2+2*th+4*(column/8) : j%2+2*th+(mixed ? 4 : 0);
        const float datum=ThreeKeys && j==3 && mixed ? 0.f : new_corr.x[index];
        const float kk=__shfl_sync(0xffffffff,datum,source);
        const float qcoef=ThreeKeys && j==3 && mixed ? qk_dot : __shfl_sync(0xffffffff,datum,source+4);
        const float factor=GateFactors ? weights[t*T+j] : expf(gt[th]-total_j);
        if (t+4*th>j) {
          const float a=bt[th]*kk*factor;
          u[th][0]-=a*previous0;
          u[th][1]-=a*previous1;
        }
        if (t+4*th>=j) {
          const float b=qcoef*factor;
          out[th][0]+=b*previous0;
          out[th][1]+=b*previous1;
        }
      }
    }
    #pragma unroll
    for (int th=0; th<TimeBlocks; ++th) {
      const int64_t token=int64_t(row)*T+t+4*th;
      const int64_t output=token*p.out.stride[0]+hv*V+v;
      const int64_t value=slot*p.values.stride[0]+dest*p.values.stride[1]+
          (position+t+4*th)*p.values.stride[2]+hv*V+v;
      write<BF>(p.out,output,out[th][0]);
      write<BF>(p.out,output+8,out[th][1]);
      write<BF>(p.values,value,u[th][0]);
      write<BF>(p.values,value+8,u[th][1]);
    }
  }
  if (blockIdx.x==0) {
    if (hv % Params::head_group == 0)
      for (int i=tid; i<T*K; i+=2*TileV)
        static_cast<BF*>(p.keys.data)[slot*p.keys.stride[0]+dest*p.keys.stride[1]+
            (position+i/K)*p.keys.stride[2]+head*K+i%K]=qk[2*(i/K)*K+i%K];
    if (tid<T)
      write<float>(p.gates,slot*p.gates.stride[0]+dest*p.gates.stride[1]+hv*L+position+tid,
                   overflow ? prefix[tid] : total[tid]);
  }
  if (p.materialize_overflow && overflow) {
    __syncwarp();
    #pragma unroll
    for (int i=lane; i<L*WarpV; i+=32) {
      const int j=i/WarpV,offset=j*TileV+local_v+i%WarpV;
      const float scaled=j<count ? float(old_u[offset])*expf(origin-history_g[j]) : 0.f;
      const BF hi=BF(scaled);
      old_u[offset]=hi;
      low_u[offset]=BF(scaled-float(hi));
    }
    __syncwarp();
    ColA hi_u,lo_u;
    load_fold_operand<true>(hi_u,old_u+local_v,TileV,lane);
    load_fold_operand<true>(lo_u,low_u+local_v,TileV,lane);
    const float decay=expf(origin);
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
        const int r=local_v+lane/4+(i%2)*8,c=k+(lane%4)*2+(i/2)*8;
        if constexpr (Split) {
          const float2 old=*reinterpret_cast<const float2*>((RawStage ? state_raw : state)+r*K+c);
          *reinterpret_cast<float2*>(state+r*K+c)=make_float2(
              old.x*decay+(hi.x[2*i]+lo.x[2*i]),old.y*decay+(hi.x[2*i+1]+lo.x[2*i+1]));
        } else {
          const uint32_t old=*reinterpret_cast<const uint32_t*>(state_hi+r*K+c);
          const float s0=__uint_as_float(old<<16),s1=__uint_as_float(old&0xffff0000u);
          *reinterpret_cast<uint32_t*>(state+r*K+c)=register_pack(
              BF(s0*decay+(hi.x[2*i]+lo.x[2*i])),BF(s1*decay+(hi.x[2*i+1]+lo.x[2*i+1])));
        }
      }
    }
  }
}
