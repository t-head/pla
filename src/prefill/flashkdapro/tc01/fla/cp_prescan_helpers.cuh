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

// Four-warp BV16/BV32 publishers, without global history/residual writes.
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
namespace fla_recurrence {
static_assert((kBv==16 || kBv==32) && kThreads==128 && kStateOrdinals==kBv/2);
static_assert(FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY && FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL);
static_assert(kPipeV+64*kBv<=kPipeScratch);
static_assert(kPipeW0+2*64*kBv<=kPipeW1); // FP32 residual, BF16 element units
static_assert(kPipeW1+4096<=kPipeK0); // swizzled residual B image
static_assert(2*(kPipeScratch+4096)<=sizeof(Stage));
static_assert(64*kBv<=4096); // output within Qg0/A slot
// CP state passes prefetch raw gate operands for the current chunk;
// recurrence_exp and __fmaf_rn remain at their original update sites.
struct CpPrefetchedDecay {
  float g[2][2];
};

CUTE_DEVICE CpPrefetchedDecay cp_prefetch_decay_inputs(
    float const* gk,int64_t last,int head) {
  CpPrefetchedDecay result;
  int key0=recurrence_state_k(threadIdx.x,0);
  int key1=recurrence_state_k(threadIdx.x,4);
  result.g[0][0]=gk[token_offset(last,head,key0)];
  result.g[0][1]=gk[token_offset(last,head,key1)];
  result.g[1][0]=gk[token_offset(last,head,64+key0)];
  result.g[1][1]=gk[token_offset(last,head,64+key1)];
  // No extra memory clobber is introduced. The compiler may sink these
  // loads; verify their placement and generated fences in PASM.
  return result;
}

template<class Accumulator>
CUTE_DEVICE void cp_update_carried_from_prefetched_decay(
    float (&carried)[2][kStateOrdinals],int panel,Accumulator const& update,
    CpPrefetchedDecay const& prefetched) {
  // Same two recurrence_exp evaluations as load_decay_pair_local, followed
  // by exactly the original ordinal selection and fused RN update order.
  float decay0=recurrence_exp(prefetched.g[panel][0]);
  float decay1=recurrence_exp(prefetched.g[panel][1]);
  CUTE_UNROLL
  for(int n=0;n<kBv/16;++n) {
    CUTE_UNROLL
    for(int value_index=0;value_index<8;++value_index) {
      int ordinal=value_index+8*n;
      float dot=update(value_index,0,n);
      float decay=((ordinal>>2)&1) ? decay1 : decay0;
      carried[panel][ordinal]=
          __fmaf_rn(carried[panel][ordinal],decay,dot);
    }
  }
}

// Preserve the native B byte image with one raw 32-bit store per lane
// and ordinal pair. Adjacent lanes exchange rounded bits, never FP values.
CUTE_DEVICE void cp_publish_state_pair32(
    uint16_t* b_storage,uint16_t* h,
    float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  static_assert((kBv==16 || kBv==32) && kThreads==128);
  int tid=threadIdx.x;
  auto* bytes=reinterpret_cast<uint8_t*>(b_storage);
  // Raw byte indices: even lanes use own.low/other.low, odd lanes use
  // other.high/own.high. Compute once, outside both publication loops.
  uint32_t selector=0x5410u ^ ((0u-uint32_t(tid&1)) & 0x6666u);
  CUTE_UNROLL
  for(int value_part=0;value_part<kBv/16;++value_part) {
    uint32_t words[4];
    CUTE_UNROLL
    for(int j=0;j<4;++j) {
      // Keep the same ordinal pair and independently round each component RN
      // once. The existing packed helper adds no intermediate FP arithmetic.
      words[j]=round_bf16_pair_packed(
          carried[panel][8*value_part+2*j],
          carried[panel][8*value_part+2*j+1]);
    }
    int owner=tid&~1;
    // The XOR by 32 keeps the two BV32 value_parts in their original slots.
    int bbase=(((owner&96)<<5)|((owner&28)<<4)|((owner<<1)&54))
        ^(32*value_part);
    CUTE_UNROLL
    for(int j=0;j<4;++j) {
      uint32_t own=words[j];
      uint32_t other=__shfl_xor_sync(0xffffffffu,own,1,32);
      // Every selector nibble is in 0..7: select raw bytes, with no
      // sign replication or floating-point conversion.
      uint32_t packed=__byte_perm(own,other,selector);
      int offset=(bbase^((j&1)<<4))+((j>>1)<<9)+((tid&1)<<3);
      *reinterpret_cast<uint32_t*>(bytes+offset)=packed;
    }
  }
}

// BEGIN EXACT EXTRACTION
CUTE_DEVICE void prescan_publish_state(
    uint16_t* b_storage,uint16_t* h,
    float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  static_assert(kBv==16 || kBv==32);
  int tid=threadIdx.x;
  CUTE_UNROLL
  for(int value_part=0;value_part<kBv/16;++value_part) {
  Bf16Pack8 packed{};
  CUTE_UNROLL
  for(int ordinal=0;ordinal<8;++ordinal) {
    uint16_t rounded=round_bf16(carried[panel][8*value_part+ordinal]);
    packed.element[ordinal]=rounded;
  }
  // Preserve the already-proved 4-warp recurrence B image for the W*state
  // MMA.  Only the history output changes to the packed register-owner image;
  // mixing in the 8-warp output reader here changes the fragment mapping.
  auto* bytes=reinterpret_cast<uint8_t*>(b_storage);
  int bbase=(((tid&96)<<5)|((tid&28)<<4)|((tid<<1)&54))^(32*value_part);
  *reinterpret_cast<uint16_t*>(bytes+bbase)=packed.element[0];
  *reinterpret_cast<uint16_t*>(bytes+bbase+8)=packed.element[1];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16))=packed.element[2];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+8)=packed.element[3];
  *reinterpret_cast<uint16_t*>(bytes+bbase+512)=packed.element[4];
  *reinterpret_cast<uint16_t*>(bytes+bbase+520)=packed.element[5];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+512)=packed.element[6];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+520)=packed.element[7];
  }
}

template<bool ZeroV,class Accumulator>
CUTE_DEVICE void prescan_publish_residual_direct_shfl(
    Stage& stage,uint16_t const* v,uint16_t* v_new,Accumulator const& residual,
    int64_t first,int head,int value_base) {
  static_assert((kBv==16 || kBv==32) && kThreads==128);
  static_assert(size<0>(Accumulator{})==8 && size<1>(Accumulator{})==1 && size<2>(Accumulator{})==kBv/16);
  static_assert(!FLA_RECURRENCE_DIRECT_V_LOAD && FLA_RECURRENCE_PACKED_RESIDUAL_CONVERT);
#if FLA_RECURRENCE_BV16
  // Preserve the established M64N16 recurrence C ownership and residual math.
  // Each four-lane group owns one row; lanes 0/1 publish BF16x8 at N=0/8.
  // This is recurrence.cu's BV16 publisher without its global v_new store.
  int lane=threadIdx.x&31;
  int source_base=lane&~3;
  auto* base=pipe_base(stage);
  auto* v_storage=base+kPipeV;
  auto* b_storage=base+kPipeW1;
  CUTE_UNROLL
  for(int output_half=0;output_half<2;++output_half) {
    float gathered[2][8];
    CUTE_UNROLL
    for(int row_half=0;row_half<2;++row_half) {
      int fragment_base=4*row_half+2*output_half;
      CUTE_UNROLL
      for(int source=0;source<4;++source) {
        gathered[row_half][source]=__shfl_sync(
            0xffffffffu,residual(fragment_base,0,0),source_base+source,32);
        gathered[row_half][source+4]=__shfl_sync(
            0xffffffffu,residual(fragment_base+1,0,0),source_base+source,32);
      }
    }
    if((lane&3)==output_half) {
      int row_base=16*(threadIdx.x>>5)+(lane>>2);
      int value=8*output_half;
      CUTE_UNROLL
      for(int row_half=0;row_half<2;++row_half) {
        int row=row_base+8*row_half;
        Bf16Pack8 output{};
        CUTE_UNROLL
        for(int i=0;i<4;++i) {
          float input_low=0.0f,input_high=0.0f;
          if constexpr(!ZeroV) {
            input_low=load_bf16(v_storage[row*kBv+value+2*i]);
            input_high=load_bf16(v_storage[row*kBv+value+2*i+1]);
          }
          float low=__fsub_rn(input_low,gathered[row_half][2*i]);
          float high=__fsub_rn(input_high,gathered[row_half][2*i+1]);
          output.pair[i]=round_bf16_pair_packed(low,high);
        }
        *reinterpret_cast<uint4*>(b_storage+swizzled(row,value))=output.vector;
      }
    }
  }
#else
  int lane=threadIdx.x&31;
  int col=8*(lane&3);
  auto* base=pipe_base(stage);
  auto* v_storage=base+kPipeV;
  auto* b_storage=base+kPipeW1;
  CUTE_UNROLL
  for(int row_half=0;row_half<2;++row_half) {
    int row=16*(threadIdx.x>>5)+(lane>>2)+8*row_half;
    uint32_t packed[4]={};
    if constexpr(!ZeroV) {
      uint4 input=*reinterpret_cast<uint4 const*>(v_storage+row*kBv+col);
      // Adjacent BF16 words -> separated pairs (i,i+4), without conversion.
      packed[0]=(input.x&0xffffu)|(input.z<<16);
      packed[1]=(input.x>>16)|(input.z&0xffff0000u);
      packed[2]=(input.y&0xffffu)|(input.w<<16);
      packed[3]=(input.y>>16)|(input.w&0xffff0000u);
      // Inverse 4x4 transpose: contiguous V reader -> exact native C owner.
      CUTE_UNROLL
      for(int stage=0;stage<2;++stage) {
        int bit=1<<stage;
        uint32_t next[4];
        CUTE_UNROLL
        for(int j=0;j<4;++j) {
          uint32_t exchanged=__shfl_xor_sync(0xffffffffu,packed[j^bit],bit,32);
          next[j]=((lane&bit)!=(j&bit)) ? exchanged : packed[j];
        }
        CUTE_UNROLL
        for(int j=0;j<4;++j) packed[j]=next[j];
      }
    }
    CUTE_UNROLL
    for(int part=0;part<2;++part) {
      CUTE_UNROLL
      for(int half=0;half<2;++half) {
        int j=2*part+half;
        float low=__fsub_rn(ZeroV ? 0.0f : load_bf16(uint16_t(packed[j])),
                           residual(4*row_half+2*half,0,part));
        float high=__fsub_rn(ZeroV ? 0.0f : load_bf16(uint16_t(packed[j]>>16)),
                            residual(4*row_half+2*half+1,0,part));
        packed[j]=round_bf16_pair_packed(low,high);
      }
    }
    // Forward transpose moves only the rounded raw BF16 results.
    CUTE_UNROLL
    for(int stage=0;stage<2;++stage) {
      int bit=1<<stage;
      uint32_t next[4];
      CUTE_UNROLL
      for(int j=0;j<4;++j) {
        uint32_t exchanged=__shfl_xor_sync(0xffffffffu,packed[j^bit],bit,32);
        next[j]=((lane&bit)!=(j&bit)) ? exchanged : packed[j];
      }
      CUTE_UNROLL
      for(int j=0;j<4;++j) packed[j]=next[j];
    }
    uint4 output;
    output.x=(packed[0]&0xffffu)|(packed[1]<<16);
    output.y=(packed[2]&0xffffu)|(packed[3]<<16);
    output.z=(packed[0]>>16)|(packed[1]&0xffff0000u);
    output.w=(packed[2]>>16)|(packed[3]&0xffff0000u);
    *reinterpret_cast<uint4*>(b_storage+swizzled(row,col))=output;
  }
#endif
}

template<bool FullRows,bool ZeroV,class Accumulator,class Coordinates>
CUTE_DEVICE void prescan_publish_residual(
    Stage& stage,uint16_t const* v,uint16_t* v_new,Accumulator const& residual,
    Coordinates const& coordinates,int64_t first,int rows,int head,
    int value_base) {
#if FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL
  if constexpr (FullRows) {
    prescan_publish_residual_direct_shfl<ZeroV>(
        stage,v,v_new,residual,first,head,value_base);
    return;
  }
#endif
  auto* base=pipe_base(stage);
  auto* residual_storage=reinterpret_cast<float*>(base+kPipeW0);
  auto* b_storage=base+kPipeW1;
  auto* v_storage=base+kPipeV;
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(residual));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(residual));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(residual));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int row=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
#if FLA_RECURRENCE_EXACT_FRAGMENT_COORDS
        residual_storage[row*kBv+value]=residual(value_index,m,n);
#else
        if(row<64 && value<kBv)
          residual_storage[row*kBv+value]=residual(value_index,m,n);
#endif
      }
  __syncthreads();
  for(int linear=8*threadIdx.x;linear<64*kBv;linear+=8*kThreads) {
    int row=linear/kBv;
    int value=linear%kBv;
    Bf16Pack8 output{};
    CUTE_UNROLL
    for(int i=0;i<8;++i) {
      uint16_t input=0;
      if constexpr(!ZeroV) {
        input=
#if FLA_RECURRENCE_PIPELINE5_AIU
          v_storage[row*kBv+value+i];
#else
          v_storage[pipe_v_element_offset(row,value+i)];
#endif
      }
      if constexpr (FullRows) {
        output.element[i]=round_bf16(__fsub_rn(
            ZeroV ? 0.0f : load_bf16(input),residual_storage[row*kBv+value+i]));
      } else {
        output.element[i]=row<rows ? round_bf16(__fsub_rn(
            ZeroV ? 0.0f : load_bf16(input),residual_storage[row*kBv+value+i])) : uint16_t(0);
      }
    }
    *reinterpret_cast<uint4*>(b_storage+swizzled(row,value))=output.vector;
  }
}

// END EXACT EXTRACTION
} // namespace fla_recurrence
#endif
