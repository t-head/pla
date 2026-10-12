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

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
namespace fla_recurrence {
template<bool Add,class BFragment,class Accumulator,class Coordinates>
CUTE_DEVICE void fused_output_tail_pretriangular(
    uint16_t* scratch,uint16_t const* a,uint16_t* output,int64_t first,
    int rows,int head,int value_base,Mma const& mma,BFragment const& residual,
    Accumulator& accumulator,Coordinates const& coordinates,
    uint16_t const* prefetched_output=nullptr) {
  // All W0 residual scratch readers have retired at the publisher barrier.
  // A has H*64 row stride, unlike the H*128 recurrence operands.
  wait_pipe_chunk();
  __syncthreads();
  // Explicit upstream raw +0 contract only for complete chunks.
  // Partial chunks retain the generic masking and publication barrier.
  if (rows < 64) {
    // Preserve every lower-triangle raw bit. Zero only upper triangle/tail.
    CUTE_UNROLL
    for(int pass=0;pass<4;++pass) {
      int linear=threadIdx.x+128*pass;
      int row=linear/8,col=8*(linear%8);
      auto* destination=reinterpret_cast<uint4*>(scratch+aiu_a_offset(row,col));
      if(row>=rows || col>row) {
        Bf16Pack8 zeros{};
        *destination=zeros.vector;
      } else if(col+7>row) {
        Bf16Pack8 values;
        values.vector=*destination;
        CUTE_UNROLL
        for(int item=0;item<8;++item)
          if(col+item>row) values.element[item]=uint16_t(0);
        *destination=values.vector;
      }
    }
    __syncthreads();
  }
  // Continue the SAME FP32 accumulator after eight Qg/H K16 atoms.
  mma_accumulate_pipe_loaded_b<AiuAReader>(scratch,mma,residual,accumulator);
  __syncthreads();
  // Correction's old output was copied bit-for-bit into the linear V slot.
  // wait0 above retires that AIU copy as well as K/A. Keep RN addition and
  // output BF16 rounding in the original publisher.
  store_fused_output_shuffle<Add,Add>(
      output,first,rows,head,value_base,accumulator,prefetched_output);
  // The next chunk's asynchronous W0 load cannot clobber this consumer.
  __syncthreads();
}

} // namespace fla_recurrence
#endif
