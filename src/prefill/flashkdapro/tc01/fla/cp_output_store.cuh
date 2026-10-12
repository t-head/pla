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

// Output-only register exchange. Scalar BF16 rounding is unchanged.
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
namespace fla_recurrence {
template<bool Add,bool Prefetched=false,class Accumulator>
CUTE_DEVICE void store_fused_output_shuffle(
    uint16_t* output,int64_t first,int rows,int head,int value_base,
    Accumulator const& accumulator,uint16_t const* prefetched_output=nullptr) {
  static_assert(!Prefetched || Add);
  static_assert(kThreads==128 && (kBv==16 || kBv==32));
  static_assert(size<0>(Accumulator{})==8 && size<1>(Accumulator{})==1 && size<2>(Accumulator{})==kBv/16);
#if FLA_RECURRENCE_BV16
  // Same physical M64N16 fragment as the established recurrence publisher.
  // Gather before the publication barrier and keep scalar BF16 rounding.
  int lane=threadIdx.x&31;
  int source_base=lane&~3;
  uint4 vectors[2];
  float corrections[2][8];
  CUTE_UNROLL
  for(int output_half=0;output_half<2;++output_half) {
    CUTE_UNROLL
    for(int row_half=0;row_half<2;++row_half) {
      int fragment_base=4*row_half+2*output_half;
      float gathered[8];
      CUTE_UNROLL
      for(int source=0;source<4;++source) {
        gathered[source]=__shfl_sync(
            0xffffffffu,accumulator(fragment_base,0,0),source_base+source,32);
        gathered[source+4]=__shfl_sync(
            0xffffffffu,accumulator(fragment_base+1,0,0),source_base+source,32);
      }
      if((lane&3)==output_half) {
        if constexpr(Add) {
          CUTE_UNROLL
          for(int item=0;item<8;++item)
            corrections[row_half][item]=gathered[item];
        } else {
          Bf16Pack8 packed;
          CUTE_UNROLL
          for(int item=0;item<8;++item)
            packed.element[item]=round_bf16(gathered[item]);
          vectors[row_half]=packed.vector;
        }
      }
    }
  }
  __syncthreads();
  if((lane&3)<2) {
    int col=8*(lane&3);
    CUTE_UNROLL
    for(int row_half=0;row_half<2;++row_half) {
      int row=16*(threadIdx.x>>5)+(lane>>2)+8*row_half;
      if(row<rows) {
        auto* destination=output+token_offset(first+row,head,value_base+col);
        if constexpr(Add) {
          Bf16Pack8 value,current;
          if constexpr(Prefetched)
            current.vector=*reinterpret_cast<uint4 const*>(
                prefetched_output+row*kBv+col);
          else
            current.vector=*reinterpret_cast<uint4 const*>(destination);
          CUTE_UNROLL
          for(int item=0;item<8;++item)
            value.element[item]=round_bf16(__fadd_rn(
                load_bf16(current.element[item]),corrections[row_half][item]));
          vectors[row_half]=value.vector;
        }
        *reinterpret_cast<uint4*>(destination)=vectors[row_half];
      }
    }
  }
#else
  int lane=threadIdx.x&31;
  uint4 vectors[2];
  float corrections[2][8];
  CUTE_UNROLL
  for(int row_half=0;row_half<2;++row_half) {
    if constexpr (Add) {
      float values[4][2];
      CUTE_UNROLL
      for(int part=0;part<2;++part) {
        CUTE_UNROLL
        for(int half=0;half<2;++half) {
          values[2*part+half][0]=accumulator(4*row_half+2*half,0,part);
          values[2*part+half][1]=accumulator(4*row_half+2*half+1,0,part);
        }
      }
      // Transpose the FP32 fragment before the final BF16 publication so the
      // correction is added directly to the existing BF16 output in FP32.
      CUTE_UNROLL
      for(int stage=0;stage<2;++stage) {
        int bit=1<<stage;
        float next[4][2];
        CUTE_UNROLL
        for(int j=0;j<4;++j) {
          CUTE_UNROLL
          for(int half=0;half<2;++half) {
            float exchanged=__shfl_xor_sync(
                0xffffffffu,values[j^bit][half],bit,32);
            next[j][half]=((lane&bit)!=(j&bit)) ? exchanged : values[j][half];
          }
        }
        CUTE_UNROLL
        for(int j=0;j<4;++j)
          CUTE_UNROLL
          for(int half=0;half<2;++half) values[j][half]=next[j][half];
      }
      CUTE_UNROLL
      for(int j=0;j<4;++j) {
        corrections[row_half][j]=values[j][0];
        corrections[row_half][4+j]=values[j][1];
      }
    } else {
      uint32_t packed[4];
      CUTE_UNROLL
      for(int part=0;part<2;++part) {
        CUTE_UNROLL
        for(int half=0;half<2;++half) {
          uint32_t lo=round_bf16(accumulator(4*row_half+2*half,0,part));
          uint32_t hi=round_bf16(accumulator(4*row_half+2*half+1,0,part));
          packed[2*part+half]=lo|(hi<<16);
        }
      }
      // Transpose 4 lanes x 4 register pairs, preserving every raw bit.
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
      vectors[row_half].x=(packed[0]&0xffffu)|(packed[1]<<16);
      vectors[row_half].y=(packed[2]&0xffffu)|(packed[3]<<16);
      vectors[row_half].z=(packed[0]>>16)|(packed[1]&0xffff0000u);
      vectors[row_half].w=(packed[2]>>16)|(packed[3]&0xffff0000u);
    }
  }
  // Retain the former output-publication barrier; no additional stage reuse.
  __syncthreads();
  int col=16*((lane>>1)&1)+8*(lane&1);
  CUTE_UNROLL
  for(int row_half=0;row_half<2;++row_half) {
    int row=16*(threadIdx.x>>5)+(lane>>2)+8*row_half;
    if(row<rows) {
      auto* destination=output+token_offset(first+row,head,value_base+col);
      if constexpr (Add) {
        Bf16Pack8 value,current;
        if constexpr(Prefetched)
          current.vector=*reinterpret_cast<uint4 const*>(
              prefetched_output+row*kBv+col);
        else
          current.vector=*reinterpret_cast<uint4 const*>(destination);
        CUTE_UNROLL
        for(int item=0;item<8;++item)
          value.element[item]=round_bf16(__fadd_rn(
              load_bf16(current.element[item]),corrections[row_half][item]));
        vectors[row_half]=value.vector;
      }
      *reinterpret_cast<uint4*>(destination)=vectors[row_half];
    }
  }
#endif
}
} // namespace fla_recurrence
#endif
