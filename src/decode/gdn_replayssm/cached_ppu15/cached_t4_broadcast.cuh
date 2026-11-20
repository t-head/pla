// SPDX-License-Identifier: Apache-2.0
// B1/H1 BF16 T4 with one checkpoint and one history slot. The consumer
// warps prepare their operands while producer warps stage the pool.
template <int TileV, int Helpers, int Unroll, bool ReuseQk, bool PrepareFold, bool Hoist, bool HvFast, bool PackedMeta, bool DelayNull, class Params>
__global__ __launch_bounds__(2*TileV+32*Helpers) void cached_t4_broadcast(Params p) {
  constexpr int K=128,V=128,L=16,T=4,WarpV=16,Warps=TileV/WarpV;
  constexpr int StateWarp=Warps, HistoryWarp=Warps+1, GateWarp=Warps+Helpers-1;
  static_assert(Helpers==2 || Helpers==3);
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  const int hv=HvFast ? blockIdx.x%4 : blockIdx.z;
  const int tile=HvFast ? blockIdx.x/4 : blockIdx.x,vbase=tile*TileV,local_v=warp*WarpV;
  const int t=lane%4,v=hv*V+vbase+local_v+lane/4;
  BF* const state=static_cast<BF*>(p.state.data)+int64_t(hv*V+vbase)*K;
  __shared__ __align__(128) BF old_k[L*K],old_u[L*TileV],lower_u[L*TileV],state_tile[TileV*K];
  __shared__ __align__(32) BF qk[2*T*K];
  __shared__ float gh[L],origin,total[T],prefix[T],beta[T],new_decay[T*T];
  __shared__ float fold_decay[PrepareFold ? L : 1];
  __shared__ int metadata[PackedMeta ? 1 : 3];

  BF val0,val1;
  if (warp<Warps) {
    const BF* input=static_cast<const BF*>(p.v.data)+t*p.v.stride[0]+v;
    val0=input[0];
    val1=input[8];
    register_normalize<T,Warps>(p,qk,0,0,warp,lane);
  } else if (warp==StateWarp) {
    cached_bulk_bf16(state_tile,state,K,TileV,K,TileV,0,K);
    __pipeline_commit();
    __pipeline_wait_prior(0);
  } else {
    const int count=static_cast<const int32_t*>(p.lengths.data)[0];
    const int bank=static_cast<const int32_t*>(p.buffers.data)[0];
    const int slot=DelayNull ? 0 : static_cast<const int32_t*>(p.slots.data)[0];
    const bool valid_metadata=unsigned(count)<=L && unsigned(bank)<2;
    asm volatile("" :: "r"(slot),"r"(count),"r"(bank) : "memory");
    const auto publish=[&](int active_slot) {
      if constexpr (PackedMeta) {
        metadata[0]=active_slot>p.null_slot && valid_metadata ? count+(bank<<5) : -1;
      } else {
        metadata[0]=active_slot;
        metadata[1]=count;
        metadata[2]=bank;
      }
    };
    if constexpr (!DelayNull)
      if (warp==GateWarp && lane==0) publish(slot);
    if (DelayNull ? valid_metadata : slot>p.null_slot) {
      if constexpr (!DelayNull) assert(slot==0 && valid_metadata);
      const bool copies=count!=0 && warp==HistoryWarp;
      if (copies) {
        cached_bulk_bf16(old_k,static_cast<const BF*>(p.keys.data)+bank*p.keys.stride[1],
                         K,count,K,L,0,p.keys.stride[2]);
        cached_bulk_bf16(old_u,static_cast<const BF*>(p.values.data)+bank*p.values.stride[1]+hv*V,
                         V,count,TileV,L,vbase,p.values.stride[2]);
        __pipeline_commit();
      }
      if (warp==GateWarp) {
        const float old=lane<count ? read<float>(p.gates,bank*p.gates.stride[1]+hv*L+lane) : 0.f;
        const float start=__shfl_sync(0xffffffff,old,count==0 ? 0 : count-1);
        const float g=lane<T ? read<float>(p.a,lane*p.a.stride[0]+hv) : 0.f;
        const float b=lane<T ? read<float>(p.b,lane*p.b.stride[0]+hv) : 0.f;
        float scan=0.f;
        #pragma unroll
        for (int j=0; j<T; ++j) {
          const float next=__shfl_sync(0xffffffff,g,j);
          if (j<=lane) scan+=next;
        }
        const float active=start+scan;
        if (lane<L) {
          gh[lane]=old;
          if constexpr (PrepareFold)
            if (count+T>L) fold_decay[lane]=lane<count ? expf(start-old) : 0.f;
        }
        if (lane<T) {
          prefix[lane]=scan;
          total[lane]=active;
          beta[lane]=b;
        }
        const float target=__shfl_sync(0xffffffff,active,(lane/4)&3);
        const float source=__shfl_sync(0xffffffff,active,lane&3);
        if (lane<T*T) new_decay[lane]=expf(target-source);
        if (lane==0) origin=start;
      }
      if (copies) __pipeline_wait_prior(0);
    }
    if constexpr (DelayNull) {
      if (warp==GateWarp) {
        const int actual=static_cast<const int32_t*>(p.slots.data)[0];
        if (actual>p.null_slot) assert(actual==0 && valid_metadata);
        if (lane==0) publish(actual);
      }
    }
  }
  __syncthreads();
  if (warp>=Warps) return;
  const int header=metadata[0];
  if (PackedMeta ? header<0 : header<=p.null_slot) return;
  const int count=PackedMeta ? header&31 : metadata[1];
  const int bank=PackedMeta ? (header>>5) : metadata[2];
  const bool has_history=count!=0,overflow=count+T>L,mixed=has_history && !overflow;

  using RowA=awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major>;
  using ColA=awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major>;
  using ColB=awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major>;
  using RowB=awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
  using Acc=awmma::fragment<awmma::accumulator,16,16,16,float>;
  Acc projection,historical,new_corr,old_corr;
  awmma::fill_fragment(projection,0.f);
  awmma::fill_fragment(historical,0.f);
  awmma::fill_fragment(new_corr,0.f);
  awmma::fill_fragment(old_corr,0.f);
  auto multiply=[&](auto type) {
    constexpr int Phase=decltype(type)::value;
    #pragma unroll Unroll
    for (int k=0; k<K; k+=16) {
      RowA query,checkpoint;
      ColB keys,current;
      if constexpr (ReuseQk) register_load_t4_pair(query,current,qk+k,lane);
      else {
        register_load_qk<true,T>(query,qk+k,lane);
        register_load_qk<false,T>(current,qk+k,lane);
      }
      load_a16(checkpoint,state_tile+local_v*K+k,K,lane);
      if (Phase==1 || (Phase<0 && mixed)) register_load_keys<T,true>(keys,qk,old_k,k,lane);
      else register_load_keys<T,false>(keys,qk,old_k,k,lane);
      awmma::mma_sync(projection,checkpoint,current,projection);
      awmma::mma_sync(new_corr,query,keys,new_corr);
      if (Phase==2 || (Phase<0 && has_history && !mixed)) {
        load_b16(keys,old_k+k,K,lane);
        awmma::mma_sync(old_corr,query,keys,old_corr);
      }
    }
  };
  if constexpr (Hoist) {
    if (mixed) multiply(std::integral_constant<int,1>{});
    else if (has_history) multiply(std::integral_constant<int,2>{});
    else multiply(std::integral_constant<int,0>{});
  } else {
    multiply(std::integral_constant<int,-1>{});
  }
  if (has_history) {
    ColA u;
    ColB coefficient;
    uint32_t words[4]{};
    const float gt=total[lane/8];
    #pragma unroll
    for (int i=0; i<2; ++i) {
      const int j=2*(lane%4)+8*i;
      const float h0=mixed ? new_corr.x[4*i] : old_corr.x[4*i];
      const float h1=mixed ? new_corr.x[4*i+1] : old_corr.x[4*i+1];
      words[i]=register_pack(j<count ? BF(h0*expf(gt-gh[j])) : BF(0.f),
                             j+1<count ? BF(h1*expf(gt-gh[j+1])) : BF(0.f));
    }
    __builtin_memcpy(coefficient.x,words,sizeof(words));
    load_fold_operand<true>(u,old_u+local_v,TileV,lane);
    awmma::mma_sync(historical,u,coefficient,historical);
  }

  const float d=expf(total[t]), b=beta[t];
  float u0=b*(float(val0)-d*projection.x[0]-historical.x[0]);
  float u1=b*(float(val1)-d*projection.x[2]-historical.x[2]);
  float o0=d*projection.x[1]+historical.x[1],o1=d*projection.x[3]+historical.x[3];
  #pragma unroll
  for (int j=0; j<T; ++j) {
    const int source=8*t+j/2+(mixed ? 2 : 0);
    const float c=new_corr.x[j%2+(mixed ? 4 : 0)];
    const float kk=__shfl_sync(0xffffffff,c,source);
    const float qk=__shfl_sync(0xffffffff,c,source+4);
    const float decay=new_decay[t*T+j];
    const float p0=__shfl_sync(0xffffffff,u0,j,4),p1=__shfl_sync(0xffffffff,u1,j,4);
    if (t>j) {
      const float a=b*kk*decay;
      u0-=a*p0;
      u1-=a*p1;
    }
    if (t>=j) {
      const float coef=qk*decay;
      o0+=coef*p0;
      o1+=coef*p1;
    }
  }
  const int dest=overflow ? 1-bank : bank,position=overflow ? 0 : count;
  const int64_t output=t*p.out.stride[0]+v;
  const int64_t value=dest*p.values.stride[1]+(position+t)*p.values.stride[2]+v;
  write<BF>(p.out,output,o0);
  write<BF>(p.out,output+8,o1);
  write<BF>(p.values,value,u0);
  write<BF>(p.values,value+8,u1);
  if (tile==0) {
    if (hv==0)
      for (int i=tid; i<T*K; i+=2*TileV)
        static_cast<BF*>(p.keys.data)[dest*p.keys.stride[1]+(position+i/K)*p.keys.stride[2]+i%K]
            =qk[2*(i/K)*K+i%K];
    if (tid<T)
      write<float>(p.gates,dest*p.gates.stride[1]+hv*L+position+tid,
                   overflow ? prefix[tid] : total[tid]);
  }

  if (p.materialize_overflow && overflow) {
    __syncwarp();
    #pragma unroll
    for (int i=lane; i<L*WarpV; i+=32) {
      const int j=i/WarpV,offset=j*TileV+local_v+i%WarpV;
      const float w=PrepareFold ? fold_decay[j] : (j<count ? expf(origin-gh[j]) : 0.f);
      const float scaled=j<count ? float(old_u[offset])*w : 0.f;
      const BF hi=BF(scaled);
      old_u[offset]=hi;
      lower_u[offset]=BF(scaled-float(hi));
    }
    __syncwarp();
    ColA uhi,ulo;
    load_fold_operand<true>(uhi,old_u+local_v,TileV,lane);
    load_fold_operand<true>(ulo,lower_u+local_v,TileV,lane);
    const float decay=expf(origin);
    #pragma unroll 1
    for (int k=0; k<K; k+=16) {
      RowB keys;
      Acc hi,lo;
      load_fold_operand<false>(keys,old_k+k,K,lane);
      awmma::fill_fragment(hi,0.f);
      awmma::fill_fragment(lo,0.f);
      awmma::mma_sync(hi,uhi,keys,hi);
      awmma::mma_sync(lo,ulo,keys,lo);
      #pragma unroll
      for (int i=0; i<4; ++i) {
        const int r=local_v+lane/4+(i%2)*8,c=k+(lane%4)*2+(i/2)*8;
        const uint32_t bits=*reinterpret_cast<const uint32_t*>(state_tile+r*K+c);
        const float s0=__uint_as_float(bits<<16),s1=__uint_as_float(bits&0xffff0000u);
        *reinterpret_cast<uint32_t*>(state+r*K+c)=register_pack(
            BF(s0*decay+(hi.x[2*i]+lo.x[2*i])),BF(s1*decay+(hi.x[2*i+1]+lo.x[2*i+1])));
      }
    }
  }
}
