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

// Independent recurrence + output candidate. No global history/residual writes.
// Included only after the frozen B-only recurrence translation unit.
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
namespace fla_recurrence {
CUTE_DEVICE void wait_fused_panel0() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.wait_group 5;\n" ::: "memory");
#endif
}
CUTE_DEVICE void wait_fused_panel1() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.wait_group 3;\n" ::: "memory");
#endif
}
// Transition has no Qg/V/A consumers. Its four nonempty groups are
// W0, W1, K0, K1; wait 3 retires W0 and wait 2 retires W1. The caller
// retains wait_pipe_chunk() before consuming either K panel. No empty
// commit groups or assumptions about empty-group counting are required.
CUTE_DEVICE void wait_transition_panel0() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.wait_group 3;\n" ::: "memory");
#endif
}
CUTE_DEVICE void wait_transition_panel1() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.wait_group 2;\n" ::: "memory");
#endif
}
CUTE_DEVICE void issue_transition_first_three(
    Stage& stage,uint16_t const* k,uint16_t const* w,
    int64_t first,int rows,int head) {
  auto* base=pipe_base(stage);
  issue_pipe_aiu_tile64<false,true>(base+kPipeW0,w,first,rows,head,0);
  issue_pipe_aiu_tile64<false,true>(base+kPipeW1,w,first,rows,head,64);
  issue_pipe_aiu_tile64<true,true>(base+kPipeK0,k,first,rows,head,0);
}
// Fixed seven-group protocol; omitting a full-row V issue would invalidate waits.
static_assert(!FLA_RECURRENCE_DIRECT_V_LOAD);
CUTE_DEVICE void issue_fused_first_six(
    Stage& stage,uint16_t* output_operand,uint16_t const* qg,
    uint16_t const* k,uint16_t const* v,uint16_t const* w,
    int64_t first,int rows,int head,int value_base) {
  auto* base=pipe_base(stage);
  issue_pipe_aiu_tile64<false,true>(base+kPipeW0,w,first,rows,head,0);
  issue_pipe_aiu_tile64<false,true>(output_operand,qg,first,rows,head,0);
  issue_pipe_aiu_tile64<false,true>(base+kPipeW1,w,first,rows,head,64);
  issue_pipe_aiu_tile64<false,true>(output_operand+4096,qg,first,rows,head,64);
  issue_pipe_aiu_v(base+kPipeV,v,first,rows,head,value_base);
  issue_pipe_aiu_tile64<true,true>(base+kPipeK0,k,first,rows,head,0);
}

static_assert(FLA_RECURRENCE_PIPELINE5_AIU && FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY);
static_assert((kBv==16 || kBv==32) && FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL);
static_assert(kPipeW0+4096<=kPipeW1 && kPipeW1+4096<=kPipeK0);

// Caller retires W0 reads with a CTA barrier before this reuse. State B is
// unchanged. The final barrier protects both the A image and the state image.
template<class Accumulator>
CUTE_DEVICE void fused_w_qg_panel(
    uint16_t* w_operand,uint16_t* scratch,uint16_t* state,
    uint16_t const* qg,int64_t first,int rows,int head,int panel,
    Mma const& mma,Accumulator& accumulator,Accumulator& output) {
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(state))));
  auto b_fragment=mma.get_thread_slice(threadIdx.x).partition_fragment_B(
      make_tensor(bp,BLogical{}));
  load_pipe_b_fragment_reference(state,mma,b_fragment);
  mma_accumulate_pipe_loaded_b<AiuAReader>(w_operand,mma,b_fragment,accumulator);
  // Both operations only read A/state; the final CTA barrier remains.
  mma_accumulate_pipe_loaded_b<AiuAReader>(scratch,mma,b_fragment,output);
  __syncthreads();
}

// Match the W/state arithmetic and state-consumer retirement of
// fused_w_qg_panel, without constructing or consuming any output operand.
template<class Accumulator>
CUTE_DEVICE void transition_w_panel(
    uint16_t* w_operand,uint16_t* state,
    Mma const& mma,Accumulator& accumulator) {
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(state))));
  auto b_fragment=mma.get_thread_slice(threadIdx.x).partition_fragment_B(
      make_tensor(bp,BLogical{}));
  load_pipe_b_fragment_reference(state,mma,b_fragment);
  mma_accumulate_pipe_loaded_b<AiuAReader>(w_operand,mma,b_fragment,accumulator);
  __syncthreads();
}

CUTE_DEVICE int aiu_a_offset(int row,int col) {
  int x=64*row+col;
  return (x&7)
      | ((((x>>3)&1)^((x>>8)&1))<<3)
      | ((((x>>5)&1)^((x>>6)&1))<<4)
      | ((((x>>4)&1)^(((x>>5)&1)&((x>>6)&1))^((x>>7)&1))<<5)
      | (((x>>8)&15)<<6)
      | (((x>>4)&3)<<10);
}
CUTE_DEVICE void issue_fused_aiu_a(
    uint16_t* destination,uint16_t const* a,int64_t first,int rows,int head) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) {
    using AiuLoad=PPU0010_AIU_LOAD<
        C<64*64*16>,bfloat16_t,false,true>;
    auto const* input=a+token_offset(first,head,0,kBt);
    AiuDesc desc;
    desc.template init<bfloat16_t,false,64,64>(
        reinterpret_cast<uint8_t const*>(input),rows,64,
        make_stride(Int<kHeads*kBt>{},_1{}));
    uint32_t output=static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    AiuLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
                  reinterpret_cast<void const*>(input),desc,0,0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  }
#else
  (void)destination;(void)a;(void)first;(void)rows;(void)head;
#endif
}
template<class BFragment,class Accumulator,class Coordinates>
CUTE_DEVICE void fused_output_tail(
    uint16_t* scratch,uint16_t const* a,uint16_t* output,int64_t first,
    int rows,int head,int value_base,Mma const& mma,BFragment const& residual,
    Accumulator& accumulator,Coordinates const& coordinates) {
  // All W0 residual scratch readers have retired at the publisher barrier.
  // A has H*64 row stride, unlike the H*128 recurrence operands.
  wait_pipe_chunk();
  __syncthreads();
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
  // Continue the SAME FP32 accumulator after eight Qg/H K16 atoms.
  mma_accumulate_pipe_loaded_b<AiuAReader>(scratch,mma,residual,accumulator);
  __syncthreads();
  // Preserve scalar RN; exchange only raw BF16 words, no output TSM image.
  store_fused_output_shuffle<false>(output,first,rows,head,value_base,accumulator);
  // The next chunk's asynchronous W0 load cannot clobber this consumer.
  __syncthreads();
}

} // namespace fla_recurrence
#endif
