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

// Exact original fused math specialized for deterministic CP state passing.
namespace fla_recurrence {
enum class CpPass { Zero, Transition, Correction, LegacyZero, LegacyRepair };
}  // namespace fla_recurrence

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
namespace fla_recurrence {
static_assert(FLA_RECURRENCE_PIPELINE5_AIU && (kBv==16 || kBv==32));
static_assert(FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY && !FLA_RECURRENCE_PRECOMPUTE_DECAY);

CUTE_DEVICE float load_cp_public_state(
    void const* initial_state,int state_is_fp32,int slot,int head,int value,int key) {
  if(!initial_state) return 0.0f;
  int64_t offset=state_offset(slot,head,value,key);
  if(state_is_fp32)
    return load_bf16(round_bf16(static_cast<float const*>(initial_state)[offset]));
  return load_bf16(static_cast<uint16_t const*>(initial_state)[offset]);
}

CUTE_DEVICE void store_cp_public_state(
    void* final_state,int state_is_fp32,int slot,int head,int value,int key,float carried) {
  if(!final_state) return;
  int64_t offset=state_offset(slot,head,value,key);
  uint16_t rounded=round_bf16(carried);
  if(state_is_fp32) static_cast<float*>(final_state)[offset]=load_bf16(rounded);
  else static_cast<uint16_t*>(final_state)[offset]=rounded;
}

template<int NtBucket,CpPass Pass>
CUTE_DEVICE void fused_pretriangular_output_body_pipeline5(
    uint16_t const* qg,uint16_t const* a,uint16_t* output,
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,float const* decay_cache,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,float* ends,float* checkpoints,
    int* executed,int segment,int segment_count,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T,Stage& stage,uint16_t* output_operand) {
  constexpr bool kPrefetchDecay=
      (Pass==CpPass::Zero || Pass==CpPass::Transition || Pass==CpPass::Correction)
      && FLA_RECURRENCE_LOCAL_DECAY && !FLA_RECURRENCE_SHFL_DECAY;
  int tid=threadIdx.x;
  int value_base=blockIdx.x*kBv;
  int head=blockIdx.y;
  int bos=cu_seqlens[0],eos=cu_seqlens[1];
  if(eos-bos != T) return;
  int nt=(T+kBt-1)/kBt;
  int64_t history_base=chunk_offsets[0];
  int slot=segment;
  int public_slot=state_indices ? state_indices[0] : 0;
  float carried[2][kStateOrdinals];
  CUTE_UNROLL
  for(int panel=0;panel<2;++panel)
    CUTE_UNROLL
    for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
      int value=recurrence_state_v(tid,ordinal);
      int key=recurrence_state_k(tid,ordinal);
      if constexpr(Pass==CpPass::Transition) {
        carried[panel][ordinal]=(value_base+value==64*panel+key) ? 1.0f : 0.0f;
      } else if constexpr(Pass==CpPass::Correction) {
        const int64_t plane=int64_t(segment_count)*FLA_HEADS*128*128;
        carried[panel][ordinal]=checkpoints[plane+state_offset(
            segment,head,value_base+value,64*panel+key)];
      } else if constexpr(Pass==CpPass::LegacyRepair) {
        carried[panel][ordinal]=ends[state_offset(
            segment-1,head,value_base+value,64*panel+key)];
      } else if constexpr(Pass==CpPass::LegacyZero) {
        carried[panel][ordinal]=segment==0
            ? __fadd_rn(load_cp_public_state(initial_state,state_is_fp32,public_slot,
                          head,value_base+value,64*panel+key),0.0f)
            : 0.0f;
      } else {
        // Segment zero directly computes its true output and C[1] from the
        // public initial state, exactly as the established single-segment path.
        // Other segments still produce independent zero-seeded outputs/B[i].
        // Thus ends[0] is C[1], while ends[i>0] retains its B[i] meaning.
        carried[panel][ordinal]=segment==0
            ? __fadd_rn(load_cp_public_state(initial_state,state_is_fp32,public_slot,
                          head,value_base+value,64*panel+key),0.0f)
            : 0.0f;
      }
    }

  Mma mma;
  auto thr=mma.get_thread_slice(tid);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},Int<kBv>{})));
  auto accumulator=Mma::make_fragment_C(coordinates);
  auto* base=pipe_base(stage);
  if(nt>0) {
    int rows0=min(kBt,eos-bos);
    if constexpr(Pass==CpPass::Transition)
      issue_transition_first_three(stage,k,w,bos,rows0,head);
    else if constexpr(Pass==CpPass::Correction)
      // The Zero kernel has published every old output before Correction.
      // Reuse the otherwise unread linear V slot to prefetch those raw bits.
      issue_fused_first_six(stage,output_operand,qg,k,output,w,bos,rows0,head,value_base);
    else
      issue_fused_first_six(stage,output_operand,qg,k,v,w,bos,rows0,head,value_base);
    issue_pipe_k1(stage,k,bos,rows0,head);
  }

  for(int chunk=0;chunk<nt;++chunk) {
    int64_t first=bos+int64_t(chunk)*kBt;
    int rows=min(kBt,eos-int(first));
    CpPrefetchedDecay prefetched_decay;
    if constexpr(kPrefetchDecay) {
      int cp_last=min(first+kBt,int64_t(eos))-1;
      prefetched_decay=cp_prefetch_decay_inputs(gk,cp_last,head);
    }
    auto* b_storage=base+kPipeScratch;
    clear(accumulator);
    auto output_accumulator=Mma::make_fragment_C(coordinates);
    clear(output_accumulator);
#if FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
    if constexpr(Pass==CpPass::Zero || Pass==CpPass::Transition || Pass==CpPass::Correction)
      cp_publish_state_pair32(
          b_storage,h,carried,0,history_base+chunk,head,value_base);
    else
      prescan_publish_state(
          b_storage,h,carried,0,history_base+chunk,head,value_base);
#elif FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
    store_pipe_history_and_state_reference(
        b_storage,h,carried,0,history_base+chunk,head,value_base);
#elif FLA_RECURRENCE_SCRATCH_HISTORY
    store_pipe_history_and_state_scratch(
        b_storage,base+kPipeHistoryScratch,h,carried,0,
        history_base+chunk,head,value_base);
#else
    store_pipe_history_and_state(
        b_storage,h,carried,0,history_base+chunk,head,value_base);
#endif
    if constexpr(Pass==CpPass::Transition) wait_transition_panel0();
    else wait_fused_panel0();
    __syncthreads();
    if constexpr(Pass==CpPass::Transition) {
      transition_w_panel(base+kPipeW0,b_storage,mma,accumulator);
    } else {
      fused_w_qg_panel(base+kPipeW0,output_operand,b_storage,qg,first,rows,head,0,
                       mma,accumulator,output_accumulator);
      issue_fused_aiu_a(output_operand,a,first,rows,head);
    }
#if FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
    if constexpr(Pass==CpPass::Zero || Pass==CpPass::Transition || Pass==CpPass::Correction)
      cp_publish_state_pair32(
          b_storage,h,carried,1,history_base+chunk,head,value_base);
    else
      prescan_publish_state(
          b_storage,h,carried,1,history_base+chunk,head,value_base);
#elif FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
    store_pipe_history_and_state_reference(
        b_storage,h,carried,1,history_base+chunk,head,value_base);
#elif FLA_RECURRENCE_SCRATCH_HISTORY
    store_pipe_history_and_state_scratch(
        b_storage,base+kPipeHistoryScratch,h,carried,1,
        history_base+chunk,head,value_base);
#else
    store_pipe_history_and_state(
        b_storage,h,carried,1,history_base+chunk,head,value_base);
#endif
    if constexpr(Pass==CpPass::Transition) wait_transition_panel1();
    else wait_fused_panel1();
    __syncthreads();
    if constexpr(Pass==CpPass::Transition)
      transition_w_panel(base+kPipeW1,b_storage,mma,accumulator);
    else
      fused_w_qg_panel(base+kPipeW1,output_operand+4096,b_storage,qg,first,rows,head,1,
                       mma,accumulator,output_accumulator);
#if FLA_RECURRENCE_FULL_ROWS_SPECIALIZATION
    if(rows==kBt) {
      prescan_publish_residual<true,Pass==CpPass::Transition || Pass==CpPass::Correction>(
          stage,v,v_new,accumulator,coordinates,first,rows,head,value_base);
    } else {
      prescan_publish_residual<false,Pass==CpPass::Transition || Pass==CpPass::Correction>(
          stage,v,v_new,accumulator,coordinates,first,rows,head,value_base);
    }
#else
    prescan_publish_residual<false,Pass==CpPass::Transition || Pass==CpPass::Correction>(
        stage,v,v_new,accumulator,coordinates,first,rows,head,value_base);
#endif
    __syncthreads();
    auto b_fragment=thr.partition_fragment_B(
        make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(
            static_cast<uintptr_t>(__cvta_generic_to_shared(base+kPipeW1)))),
            BLogical{}));
    load_pipe_b_fragment(base+kPipeW1,mma,b_fragment);
#if FLA_RECURRENCE_SKIP_BFRAGMENT_BARRIER
#elif FLA_RECURRENCE_SKIP_LONG_BFRAGMENT_BARRIER
    if constexpr (NtBucket!=2) __syncthreads();
#else
    __syncthreads();
#endif

    if constexpr(Pass==CpPass::Transition) {
      wait_pipe_chunk();
      __syncthreads();
    } else {
      fused_output_tail_pretriangular<Pass==CpPass::Correction>(
          output_operand,a,output,first,rows,head,value_base,
          mma,b_fragment,output_accumulator,coordinates,
          Pass==CpPass::Correction ? base+kPipeV : nullptr);
    }
    int last=min(first+kBt,int64_t(eos))-1;
    clear(accumulator);
#if FLA_RECURRENCE_PIPELINE5_AIU
    mma_accumulate_pipe_loaded_b<AiuATransReader>(
        base+kPipeK0,mma,b_fragment,accumulator);
#else
    mma_accumulate_pipe_k(base+kPipeK0,mma,b_fragment,accumulator);
#endif
    if constexpr(kPrefetchDecay)
      cp_update_carried_from_prefetched_decay(
          carried,0,accumulator,prefetched_decay);
    else
      update_carried_from_fragment<
          FLA_RECURRENCE_PRECOMPUTE_DECAY && NtBucket==2>(
          carried,0,accumulator,gk,decay_cache,chunk,last,head);
    __syncthreads();
    bool has_next=chunk+1<nt;
    int64_t next_first=first+kBt;
    int next_rows=has_next ? min(kBt,eos-int(next_first)) : 0;
    if(has_next) {
      if constexpr(Pass==CpPass::Transition)
        issue_transition_first_three(stage,k,w,next_first,next_rows,head);
      else if constexpr(Pass==CpPass::Correction)
        issue_fused_first_six(stage,output_operand,qg,k,output,w,next_first,next_rows,head,value_base);
      else
        issue_fused_first_six(stage,output_operand,qg,k,v,w,next_first,next_rows,head,value_base);
    }

    clear(accumulator);
#if FLA_RECURRENCE_PIPELINE5_AIU
    mma_accumulate_pipe_loaded_b<AiuATransReader>(
        base+kPipeK1,mma,b_fragment,accumulator);
#else
    mma_accumulate_pipe_k(base+kPipeK1,mma,b_fragment,accumulator);
#endif
    if constexpr(kPrefetchDecay)
      cp_update_carried_from_prefetched_decay(
          carried,1,accumulator,prefetched_decay);
    else
      update_carried_from_fragment<
          FLA_RECURRENCE_PRECOMPUTE_DECAY && NtBucket==2>(
          carried,1,accumulator,gk,decay_cache,chunk,last,head);
    __syncthreads();
    if(has_next) issue_pipe_k1(stage,k,next_first,next_rows,head);
    if constexpr(Pass==CpPass::LegacyZero || Pass==CpPass::LegacyRepair) {
      if(((chunk==1 && nt>2) || (chunk==3 && nt>4)) && segment>0) {
        float* checkpoint_plane=checkpoints+(chunk==3
            ? int64_t(segment_count)*FLA_HEADS*128*128 : 0);
        if constexpr(Pass==CpPass::LegacyZero) {
          CUTE_UNROLL
          for(int panel=0;panel<2;++panel)
            CUTE_UNROLL
            for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
              int value=recurrence_state_v(tid,ordinal);
              int key=recurrence_state_k(tid,ordinal);
              checkpoint_plane[state_offset(
                  segment,head,value_base+value,64*panel+key)]=
                  carried[panel][ordinal];
            }
        } else {
          unsigned mismatch=0;
          CUTE_UNROLL
          for(int panel=0;panel<2;++panel)
            CUTE_UNROLL
            for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
              int value=recurrence_state_v(tid,ordinal);
              int key=recurrence_state_k(tid,ordinal);
              unsigned lhs=__float_as_uint(carried[panel][ordinal]);
              unsigned rhs=__float_as_uint(checkpoint_plane[state_offset(
                  segment,head,value_base+value,64*panel+key)]);
              mismatch |= (lhs!=rhs) | ((lhs&0x7fffffffu)>=0x7f800000u)
                                     | ((rhs&0x7fffffffu)>=0x7f800000u);
            }
          auto* flags=reinterpret_cast<unsigned*>(base+kPipeScratch);
          unsigned warp_bad=__any_sync(0xffffffffu,mismatch!=0);
          if((tid&31)==0) flags[tid>>5]=warp_bad;
          __syncthreads();
          bool equal=(flags[0]|flags[1]|flags[2]|flags[3])==0;
          __syncthreads();
          if(equal) {
            if(tid==0) {
              CUTE_UNROLL
              for(int part=0;part<kBv/16;++part)
                executed[(segment*FLA_HEADS+head)*8+(kBv/16)*blockIdx.x+part]=chunk+1;
            }
            wait_pipe_chunk();
            __syncthreads();
            if(segment==segment_count-1 && final_state) {
              CUTE_UNROLL
              for(int panel=0;panel<2;++panel)
                CUTE_UNROLL
                for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
                  int value=recurrence_state_v(tid,ordinal);
                  int key=recurrence_state_k(tid,ordinal);
                  float value_at_end=ends[state_offset(
                      segment,head,value_base+value,64*panel+key)];
                  store_cp_public_state(final_state,state_is_fp32,public_slot,
                      head,value_base+value,64*panel+key,value_at_end);
                }
            }
            return;
          }
        }
      }
    }
  }

  if constexpr(Pass==CpPass::LegacyZero || Pass==CpPass::LegacyRepair) {
    CUTE_UNROLL
    for(int panel=0;panel<2;++panel)
      CUTE_UNROLL
      for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
        int value=recurrence_state_v(tid,ordinal);
        int key=recurrence_state_k(tid,ordinal);
        float value_at_end=carried[panel][ordinal];
        ends[state_offset(slot,head,value_base+value,64*panel+key)]=value_at_end;
        if((Pass==CpPass::LegacyZero && segment_count==1) ||
           (Pass==CpPass::LegacyRepair && segment==segment_count-1))
          store_cp_public_state(final_state,state_is_fp32,public_slot,head,
              value_base+value,64*panel+key,value_at_end);
      }
    if(tid==0) {
      CUTE_UNROLL
      for(int part=0;part<kBv/16;++part)
        executed[(segment*FLA_HEADS+head)*8+(kBv/16)*blockIdx.x+part]=nt;
    }
  } else if constexpr(Pass==CpPass::Correction) {
    // Zero has completed all initial-state reads before this kernel starts,
    // including when initial_state and final_state alias the same pool slot.
    // The homogeneous response already in carried replaces the unused final
    // transition/prefix product; retain one RN addition and the public BF16
    // rounding contract. This changes the final-state numerical association.
    if(segment==segment_count-1 && final_state) {
      CUTE_UNROLL
      for(int panel=0;panel<2;++panel)
        CUTE_UNROLL
        for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
          int value=value_base+recurrence_state_v(tid,ordinal);
          int key=64*panel+recurrence_state_k(tid,ordinal);
          float bias=ends[state_offset(segment,head,value,key)];
          float value_at_end=__fadd_rn(carried[panel][ordinal],bias);
          store_cp_public_state(final_state,state_is_fp32,public_slot,head,
              value,key,value_at_end);
        }
    }
  } else {
    float* destination=Pass==CpPass::Transition ? checkpoints : ends;
    CUTE_UNROLL
    for(int panel=0;panel<2;++panel)
      CUTE_UNROLL
      for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
        int value=recurrence_state_v(tid,ordinal);
        int key=recurrence_state_k(tid,ordinal);
        float value_at_end=carried[panel][ordinal];
        destination[state_offset(slot,head,value_base+value,64*panel+key)]=value_at_end;
        if constexpr(Pass==CpPass::Zero)
          if(segment_count==1)
            store_cp_public_state(final_state,state_is_fp32,public_slot,head,
                value_base+value,64*panel+key,value_at_end);
      }
  }
}
} // namespace fla_recurrence
#endif
