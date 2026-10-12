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

#include "heads.h"
#include "fla_activation.h"
// Frozen FLA standalone recompute, N1/H8/D128/BT64, CuTe C++ tc01 baseline.
// Private qualification path; not connected to the production dispatcher.
#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#if !defined(__HGGCCC__)
#error "Use the frozen tc01 SDK compiler"
#endif
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/ppu_tensor_mix.hpp>
#include <cute/arch/mma_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>

namespace cute {
template<bool Trans> struct FlaRecomputeRawReader {};
template<bool Trans> struct Copy_Traits<FlaRecomputeRawReader<Trans>> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32,_128>,Stride<_128,_1>>;
  using DstLayout = Layout<Shape<_32,Shape<_32,_4>>,Stride<_32,Stride<_1,_1024>>>;
  using RefLayout = DstLayout;
  template<class TS,class SL,class TD,class DL>
  CUTE_HOST_DEVICE friend void copy_unpack(Copy_Traits const&,
      Tensor<TS,SL> const& src,Tensor<TD,DL>& dst) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    static_assert(is_mix_iterator<typename TS::iterator>::value);
    auto coord=src.data().coord_;
    int row=Trans ? int(get<0>(coord)) : int(get<1>(coord));
    int col=Trans ? int(get<1>(coord)) : int(get<0>(coord));
    uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(src.data().ptr_.get()));
    uint32_t address=base+2*(64*row+(col^(8*(row%8))));
    auto* words=reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(dst.data()));
    if constexpr (Trans) {
      asm volatile("ppu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]) : "r"(address));
    } else {
      asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3]) : "r"(address));
    }
#else
    CUTE_INVALID_CONTROL_PATH("FLA recompute requires the tc01 device image");
#endif
  }
};
} // namespace cute

namespace fla_recompute_h8 {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
using namespace cute;
using Mma=decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{},
    Layout<Shape<_4,_2,_1>,Stride<_2,_1,_8>>{}));
using AReader=Copy_Atom<PPU0010_TSM_LD_SWZL<bfloat16_t,64,64,false,false,1>,bfloat16_t>;
using BReader=Copy_Atom<FlaRecomputeRawReader<true>,bfloat16_t>;
using ALogical=Layout<Shape<_64,_64>,Stride<_64,_1>>;
using BLogical=Layout<Shape<_64,_64>,Stride<_1,_64>>;

union Pair { uint32_t word; __ppu_bfloat162 bf; };
union Eight { uint4 vector; uint32_t word[4]; };

CUTE_DEVICE uint32_t round_pair(float a,float b) {
  Pair value; value.bf=__floats2bfloat162_rn(a,b); return value.word;
}
CUTE_DEVICE float split_exp(float x) {
  float y=__fmul_rn(x,__uint_as_float(0x3fb8aa3bu));
  bool low=y < -126.0f;
  float core=__exp2f(__fmul_rn(y,low ? 0.5f : 1.0f));
  return __fmul_rn(core,low ? core : 1.0f);
}
CUTE_DEVICE void copy16(void* dest,void const* source,bool valid) {
  uint32_t shared=static_cast<uint32_t>(__cvta_generic_to_shared(dest));
  asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(shared),"l"(source),"r"(valid ? 16 : 0) : "memory");
}
CUTE_DEVICE void commit() { asm volatile("ppu.cp.async.commit_group;" ::: "memory"); }
template<int N> CUTE_DEVICE void wait() {
  asm volatile("ppu.cp.async.wait_group %0;" :: "n"(N) : "memory");
}
CUTE_DEVICE int swizzled(int r,int c) { return 64*r+(c^(8*(r%8))); }
// A-only: native 64x64 BF16 cube; B remains legacy.
CUTE_DEVICE int aiu_a_offset(int row,int col) {
  int x=64*row+col;
  return (x&7)
      | ((((x>>3)&1)^((x>>8)&1))<<3)
      | ((((x>>5)&1)^((x>>6)&1))<<4)
      | ((((x>>4)&1)^(((x>>5)&1)&((x>>6)&1))^((x>>7)&1))<<5)
      | (((x>>8)&15)<<6)
      | (((x>>4)&3)<<10);
}
CUTE_DEVICE int64_t offset(int64_t token,int head,int feature,int width=128) {
  return (token*FLA_DEVICE_HEADS(y)+head)*width+feature;
}

// Every lane publishes two vectors, including zero-sized masked copies. A
// valid in-allocation source address is used for drain/invalid rows: cp.async
// has source size0 then, so the mathematical/pipeline operation is unchanged.
CUTE_DEVICE void prefetch_bf16(unsigned char* target,uint16_t const* input,
    int64_t first,int rows,int head,int panel,bool drain=false) {
  int tid=threadIdx.x,c=8*(tid%8);
  CUTE_UNROLL
  for(int h=0;h<2;++h) {
    int r=tid/8+32*h;
    bool valid=!drain && r<rows;
    int64_t source=valid ? offset(first+r,head,64*panel+c) : offset(first,head,0);
    copy16(target+128*r+2*c,input+source,valid);
  }
  commit();
}
CUTE_DEVICE void prefetch_gate(unsigned char* target,float const* input,
    int64_t first,int rows,int head,int panel,bool drain=false) {
  int tid=threadIdx.x,c=8*(tid%8);
  CUTE_UNROLL
  for(int h=0;h<2;++h) {
    int r=tid/8+32*h;
    bool valid=!drain && r<rows;
    CUTE_UNROLL
    for(int j=0;j<2;++j) {
      int64_t source=valid ? offset(first+r,head,64*panel+c+4*j) : offset(first,head,0);
      copy16(target+256*r+4*c+16*j,input+source,valid);
    }
  }
  commit();
}

template<class Pointer>
CUTE_DEVICE auto b_source_slice(Pointer pointer,int tid,int f,int q) {
  int warp=tid/32,lane=tid%32;
  auto iterator=make_ptr_inttuple_mix_iter(
      make_coord(16*q+lane%8+8*((lane>>4)&1),32*(warp%2)+16*f+(lane&8)),pointer);
  return make_tensor(iterator,make_layout(make_shape(_8{})));
}

// These exact byte permutations are frozen LLIR %165..207. They exchange
// eight noncontiguous MMA values into each producer's contiguous eight BF16s.
CUTE_DEVICE int exchange_write_base(int t) {
  return (((t<<2)&816) | ((t&1) ? 2112 : 0)) ^ ((t&32)<<1)
      | ((t&2) ? 4104 : 0) | ((t&16) ? 1028 : 0);
}
CUTE_DEVICE int exchange_read_base(int t) {
  return (((t<<1)&440) | ((t&1)<<2)) ^ ((t&32) ? 1028 : 0)
      | ((t&2)<<5);
}

CUTE_DEVICE void dot_and_store(unsigned char* a_base,unsigned char* b_base,
    uint16_t* output,int64_t first,int rows,int head,int panel) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(static_cast<uintptr_t>(__cvta_generic_to_shared(a_base))));
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(static_cast<uintptr_t>(__cvta_generic_to_shared(b_base))));
  auto a=make_tensor(ap,ALogical{});
  auto b=make_tensor(bp,BLogical{});
  Mma mma;
  auto thr=mma.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  auto r_b=thr.partition_fragment_B(b);
  auto copy_a=make_tiled_copy_A(AReader{},mma);
  auto copy_b=make_tiled_copy_B(BReader{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  auto d_b=copy_b.get_thread_slice(tid).retile_D(r_b);
  auto coords=thr.partition_C(make_identity_tensor(make_shape(_64{},_64{})));
  auto accum=Mma::make_fragment_C(coords);
  clear(accum);
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    CUTE_UNROLL
    for(int f=0;f<2;++f) {
      auto src=b_source_slice(bp,tid,f,q);
      auto dst=d_b(_,f,q);
      BReader{}.call(src,dst);
    }
    cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accum);
  }
  // Warp-local BF16 exchange; never overwrite B storage.
  // The caller's next wait/barrier protects B before its next publication.
  // Each quad transposes 4 registers x 4 lanes; both shuffles are warp-wide.
  int lane=tid%32,warp=tid/32;
  CUTE_UNROLL
  for(int h=0;h<2;++h) {
    uint32_t x[4];
    CUTE_UNROLL
    for(int p=0;p<4;++p) {
      int src_f=p/2,src_p=2*h+(p%2);
      x[p]=round_pair(accum(2*src_p,0,src_f),accum(2*src_p+1,0,src_f));
    }
    CUTE_UNROLL
    for(int bit=1;bit<=2;bit*=2) {
      CUTE_UNROLL
      for(int first_reg=0;first_reg<4;++first_reg) {
        if((first_reg&bit)==0) {
          int second_reg=first_reg|bit;
          uint32_t send=(lane&bit) ? x[first_reg] : x[second_reg];
          uint32_t recv=__shfl_xor_sync(0xffffffffu,send,bit,32);
          if(lane&bit) x[first_reg]=recv;
          else x[second_reg]=recv;
        }
      }
    }
    Eight store;
    store.word[0]=__byte_perm(x[0],x[1],0x5410u);
    store.word[1]=__byte_perm(x[2],x[3],0x5410u);
    store.word[2]=__byte_perm(x[0],x[1],0x7632u);
    store.word[3]=__byte_perm(x[2],x[3],0x7632u);
    int r=16*(warp/2)+lane/4+8*h;
    int c=32*(warp%2)+16*((lane/2)%2)+8*(lane%2);
    if(r<rows) *reinterpret_cast<uint4*>(output+offset(first+r,head,64*panel+c))=store.vector;
  }
}

CUTE_DEVICE void publish_u(unsigned char* raw,unsigned char* rounded,float const (&beta)[2]) {
  int tid=threadIdx.x,c=8*(tid%8);
  CUTE_UNROLL
  for(int h=0;h<2;++h) {
    int r=tid/8+32*h;
    Eight data; data.vector=*reinterpret_cast<uint4*>(raw+128*r+2*c);
    CUTE_UNROLL
    for(int p=0;p<4;++p) {
      Pair pair; pair.word=data.word[p];
      float2 value=__bfloat1622float2(pair.bf);
      data.word[p]=round_pair(__fmul_rn(value.x,beta[h]),__fmul_rn(value.y,beta[h]));
    }
    *reinterpret_cast<uint4*>(rounded+2*swizzled(r,c))=data.vector;
  }
  __syncthreads();
}

CUTE_DEVICE void publish_w_kg(unsigned char* raw_k,unsigned char* raw_g,
    unsigned char* rounded,float const (&beta)[2],float const* gate,uint16_t* kg,
    int64_t first,int rows,int head,int panel) {
  int tid=threadIdx.x,c=8*(tid%8);
  float last[8];
  CUTE_UNROLL
  for(int j=0;j<8;++j) last[j]=gate[offset(first+rows-1,head,64*panel+c+j)];
  CUTE_UNROLL
  for(int h=0;h<2;++h) {
    int r=tid/8+32*h;
    Eight k,scaled,kg_value; k.vector=*reinterpret_cast<uint4*>(raw_k+128*r+2*c);
    auto const* g=reinterpret_cast<float const*>(raw_g)+64*r+c;
    CUTE_UNROLL
    for(int p=0;p<4;++p) {
      Pair pair; pair.word=k.word[p];
      float2 value=__bfloat1622float2(pair.bf);
      float gx=g[2*p],gy=g[2*p+1];
      // Keep original K for Kg; no BF16 rounding at the k*beta boundary.
      float wx=__fmul_rn(__fmul_rn(value.x,beta[h]),split_exp(gx));
      float wy=__fmul_rn(__fmul_rn(value.y,beta[h]),split_exp(gy));
      scaled.word[p]=round_pair(wx,wy);
      kg_value.word[p]=round_pair(
          __fmul_rn(value.x,split_exp(__fsub_rn(last[2*p],gx))),
          __fmul_rn(value.y,split_exp(__fsub_rn(last[2*p+1],gy))));
    }
    *reinterpret_cast<uint4*>(rounded+2*swizzled(r,c))=scaled.vector;
    if(r<rows) *reinterpret_cast<uint4*>(kg+offset(first+r,head,64*panel+c))=kg_value.vector;
  }
  __syncthreads();
}
#endif
} // namespace fla_recompute_h8

extern "C" __global__ __launch_bounds__(256,1) __block_size__((256,1,1))
void fla_recompute_h8_bf16_d128_bt64(uint16_t const* k,uint16_t const* v,
    uint16_t const* beta_raw,int64_t beta_stride,uint16_t const* A,float const* g,
    uint16_t* w,uint16_t* u,uint16_t* kg,int32_t const* cu,
    int32_t const* indices,int total_tokens,int n) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  using namespace fla_recompute_h8;
  extern __shared__ __align__(128) unsigned char shared[];
  int tid=threadIdx.x,head=blockIdx.y;
  int sequence=indices[2*blockIdx.x],chunk=indices[2*blockIdx.x+1];
  if(sequence<0 || sequence>=n || chunk<0) return;
  int bos=cu[sequence],eos=cu[sequence+1];
  int64_t first=int64_t(bos)+int64_t(chunk)*64;
  if(bos<0 || eos>total_tokens || first>=eos) return;
  int remaining=eos-int(first);
  int rows=remaining<64 ? remaining : 64;
  prefetch_bf16(shared,v,first,rows,head,0);
  float b[2];
  CUTE_UNROLL
  for(int h=0;h<2;++h) {
    int r=tid/8+32*h,c=8*(tid%8);
    b[h]=r<rows ? fla_sigmoid_bf16(beta_raw[(first+r)*beta_stride+head]) : 0.0f;
    uint4 data=make_uint4(0,0,0,0);
    if(r<rows) data=*reinterpret_cast<uint4 const*>(A+offset(first+r,head,c,64));
    *reinterpret_cast<uint4*>(shared+49152+2*aiu_a_offset(r,c))=data;
  }
  // U: two raw V buffers, one rounded-B/output exchange buffer. A stays live.
  __syncthreads();
  prefetch_bf16(shared+8192,v,first,rows,head,1);
  CUTE_UNROLL
  for(int panel=0;panel<2;++panel) {
    wait<1>(); __syncthreads();
    publish_u(shared+8192*panel,shared+16384,b);
    dot_and_store(shared+49152,shared+16384,u,first,rows,head,panel);
    prefetch_bf16(shared+8192*panel,v,first,rows,head,panel,true);
  }
  wait<0>(); __syncthreads(); // No outstanding V writes when G aliases low TSM.
  // W: K0/G0/K1/G1 groups; both raw G panels consume the former U region.
  prefetch_bf16(shared+32768,k,first,rows,head,0);
  prefetch_gate(shared,g,first,rows,head,0);
  __syncthreads();
  prefetch_bf16(shared+40960,k,first,rows,head,1);
  prefetch_gate(shared+16384,g,first,rows,head,1);
  #pragma unroll 1
  for(int panel=0;panel<2;++panel) {
    wait<2>(); __syncthreads();
    publish_w_kg(shared+32768+8192*panel,shared+16384*panel,shared+57344,b,
                 g,kg,first,rows,head,panel);
    dot_and_store(shared+49152,shared+57344,w,first,rows,head,panel);
    prefetch_bf16(shared+32768+8192*panel,k,first,rows,head,panel,true);
    prefetch_gate(shared+16384*panel,g,first,rows,head,panel,true);
  }
  wait<0>(); __syncthreads();
#else
  __trap();
#endif
}

// Prepare-only opt-in. Never called from the graph-captured run/launch path.
extern "C" hggcError_t configure_fla_recompute_h8_bf16_d128_bt64() {
  return hggcFuncSetAttribute(fla_recompute_h8_bf16_d128_bt64,
      hggcFuncAttributeMaxDynamicSharedMemorySize,65536);
}
extern "C" hggcError_t launch_fla_recompute_h8_bf16_d128_bt64(
    void const* k,void const* v,uint16_t const* beta_raw,int64_t beta_stride,
    void const* A,float const* g,
    void* w,void* u,void* kg,int32_t const* cu,int32_t const* indices,
    int chunks,int total_tokens,int n,int heads,hggcStream_t stream) {
  if(chunks==0) return hggcSuccess;
  fla_recompute_h8_bf16_d128_bt64<<<dim3(chunks,heads),256,65536,stream>>>(
      static_cast<uint16_t const*>(k),static_cast<uint16_t const*>(v),beta_raw,beta_stride,
      static_cast<uint16_t const*>(A),g,static_cast<uint16_t*>(w),
      static_cast<uint16_t*>(u),static_cast<uint16_t*>(kg),cu,indices,total_tokens,n);
  return hggcGetLastError();
}
