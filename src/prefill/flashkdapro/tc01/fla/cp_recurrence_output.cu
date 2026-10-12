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

// Monolithic tc01 FLA context-parallel recurrence/output stage.
//
// Reuse the target tree's current recurrence numerical body, but suppress its
// serial external kernels.  The CP build uses BV32 and emits only the
// deterministic state-passing symbols below, so it cannot shadow the shipped BV16 path.
// A separately named state-only BV16 unit serves small work grids.
#define FLA_RECURRENCE_BODY_ONLY 1
#include <type_traits>
#include "../../kernels/flashkda_ppu_util.cuh"
#include "recurrence.cu"
#include "cp_prescan_helpers.cuh"
#include "cp_output_store.cuh"
#include "cp_fused_helpers.cuh"
#include "cp_pretriangular_tail.cuh"
#include "cp_pretriangular_body.cuh"
#include "cp_value_slice.h"

#if !defined(FLA_CP_BV16_VARIANT)
extern "C" hggcError_t FLA_JOIN_HEAD(configure_fla_cp_pretriangular_output_bv16, FLA_HEADS)();
extern "C" hggcError_t FLA_JOIN_HEAD(launch_fla_cp_pretriangular_output_bv16, FLA_HEADS)(
    void const*,void const*,void*,void const*,void const*,void const*,
    float const*,void const*,void*,int,int32_t const*,float*,float*,int*,
    int32_t const*,int64_t const*,int,int,hggcStream_t);
#endif

template<fla_recurrence::CpPass Pass>
__global__ __launch_bounds__(128,1) __block_size__((128,1,1))
void fla_cp_pretriangular_output(
    uint16_t const* qg,uint16_t const* a,uint16_t* output,
    uint16_t const* kg,uint16_t const* u,uint16_t const* w,
    float const* gate,void const* initial_state,void* final_state,
    int state_is_fp32,int32_t const* state_indices,float* ends,
    float* checkpoints,int* executed,int32_t const* bounds,
    int64_t const* offsets,int repair_segment,int segment_count) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  // Transition does not consume Qg/A or produce output.
  __shared__ __align__(128) uint16_t output_operand[
      Pass==fla_recurrence::CpPass::Transition ? 1 : 8192];
  extern __shared__ __align__(128) unsigned char storage[];
  auto& stage=*reinterpret_cast<fla_recurrence::Stage*>(storage);
  int segment=Pass==fla_recurrence::CpPass::LegacyRepair
      ? repair_segment : int(blockIdx.z);
  // Zero owns segment 0's true recurrence. Only segments 1..S-2 need M[i],
  // and only segments 1..S-1 need an output correction. Legacy grids retain
  // their original segment numbering; these shifted grids are state-only.
  if constexpr(Pass==fla_recurrence::CpPass::Transition ||
               Pass==fla_recurrence::CpPass::Correction)
    ++segment;
  int t=bounds[segment+1]-bounds[segment],nt=(t+63)/64;
  if(nt<=32)
    fla_recurrence::fused_pretriangular_output_body_pipeline5<0,Pass>(
        qg,a,output,kg,u,w,nullptr,gate,nullptr,nullptr,initial_state,
        final_state,state_is_fp32,state_indices,ends,checkpoints,executed,
        segment,segment_count,bounds+segment,offsets+segment,t,stage,
        output_operand);
  else if(nt<=128)
    fla_recurrence::fused_pretriangular_output_body_pipeline5<1,Pass>(
        qg,a,output,kg,u,w,nullptr,gate,nullptr,nullptr,initial_state,
        final_state,state_is_fp32,state_indices,ends,checkpoints,executed,
        segment,segment_count,bounds+segment,offsets+segment,t,stage,
        output_operand);
  else
    fla_recurrence::fused_pretriangular_output_body_pipeline5<2,Pass>(
        qg,a,output,kg,u,w,nullptr,gate,nullptr,nullptr,initial_state,
        final_state,state_is_fp32,state_indices,ends,checkpoints,executed,
        segment,segment_count,bounds+segment,offsets+segment,t,stage,
        output_operand);
#else
  __trap();
#endif
}

namespace {
constexpr int kCpPrefixRows = 16;
constexpr int kCpPrefixThreads = 256;
constexpr int kCpStateElements = 128 * 128;

struct alignas(128) CpPrefixStage {
  uint16_t carry[kCpPrefixRows * 128];
  uint16_t transition[kCpStateElements];
};
static_assert(sizeof(CpPrefixStage) == 36864);

template<int Heads>
__global__ __launch_bounds__(kCpPrefixThreads,1)
void fla_cp_state_prefix(
    float const* bias,float const* transition,float* carries,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int segment_count) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  __shared__ CpPrefixStage stage;
  int tid=int(threadIdx.x);
  int head=int(blockIdx.y);
  int value_base=int(blockIdx.x)*kCpPrefixRows;
  constexpr int kCarryElements=kCpPrefixRows*128;
  const int64_t plane=int64_t(segment_count)*Heads*kCpStateElements;

  for(int linear=tid;linear<kCarryElements;linear+=kCpPrefixThreads) {
    int value=value_base+linear/128;
    int key=linear%128;
    // Zero already folded the public initial state into segment zero's true
    // recurrence: bias[0] is C[1], unlike zero-seeded B[i] for every i>0.
    // C[0] and M[0] are neither produced nor consumed by this prefix.
    float current=bias[fla_recurrence::state_offset(0,head,value,key)];
    stage.carry[linear]=fla_recurrence::round_bf16(current);
    carries[plane+fla_recurrence::state_offset(1,head,value,key)]=current;
  }
  __syncthreads();

  // Only incoming carries C[1..S-1] are consumed by correction. Propagate
  // through the middle segments; the last correction publishes public final.
  // For S=2 the seed above is sufficient, with no transition/prefix MMA.
  for(int segment=1;segment+1<segment_count;++segment) {
    // The workspace plane and each state row are 16-byte aligned. Publish
    // four adjacent entries per thread: identical scalar BF16 RN conversions,
    // fewer global-load dependency rounds, and one raw 8-byte shared store.
    // The following CTA barrier and every consumer/MMA order are unchanged.
    static_assert(kCpStateElements % (4*kCpPrefixThreads) == 0);
    static_assert(kCpStateElements % 4 == 0);
    for(int linear=4*tid;linear<kCpStateElements;linear+=4*kCpPrefixThreads) {
      auto const* source=transition+fla_recurrence::state_offset(
          segment,head,linear/128,linear%128);
      float4 value=*reinterpret_cast<float4 const*>(source);
      uint2 packed;
      packed.x=uint32_t(fla_recurrence::round_bf16(value.x))
          | (uint32_t(fla_recurrence::round_bf16(value.y))<<16);
      packed.y=uint32_t(fla_recurrence::round_bf16(value.z))
          | (uint32_t(fla_recurrence::round_bf16(value.w))<<16);
      *reinterpret_cast<uint2*>(stage.transition+linear)=packed;
    }
    __syncthreads();

    int warp=tid/32,lane=tid%32;
    int key_block=warp;
    auto carry_tensor=cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<flashkda::BF16*>(stage.carry)),
        cute::make_layout(cute::make_shape(cute::Int<kCpPrefixRows>{},cute::_128{}),
                          cute::make_stride(cute::_128{},cute::_1{})));
    auto transition_tensor=cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<flashkda::BF16*>(stage.transition)),
        cute::make_layout(cute::make_shape(cute::_128{},cute::_128{}),
                          cute::make_stride(cute::_1{},cute::_128{})));
    auto accumulator=flashkda::make_acc16_bf16_1warp(lane);
    cute::clear(accumulator);
    CUTE_UNROLL
    for(int key_chunk=0;key_chunk<8;++key_chunk) {
      auto a_tile=cute::local_tile(carry_tensor,
          cute::make_shape(cute::_16{},cute::_16{}),
          cute::make_coord(cute::_0{},key_chunk));
      auto b_tile=cute::local_tile(transition_tensor,
          cute::make_shape(cute::_16{},cute::_16{}),
          cute::make_coord(key_block,key_chunk));
      flashkda::gemm16_ss_1warp_bf16_acc(a_tile,b_tile,lane,accumulator);
    }

    flashkda::TiledMma1WarpBf16 mma;
    auto coordinates=mma.get_thread_slice(lane).partition_C(
        cute::make_identity_tensor(cute::make_shape(cute::_16{},cute::_16{})));
    auto result=cute::recast<float>(accumulator);
    CUTE_UNROLL
    for(int m=0;m<int(cute::size<1>(coordinates));++m)
      CUTE_UNROLL
      for(int n=0;n<int(cute::size<2>(coordinates));++n)
        CUTE_UNROLL
        for(int item=0;item<int(cute::size<0>(coordinates));++item) {
          auto coordinate=coordinates(item,m,n);
          int value=value_base+int(cute::get<0>(coordinate));
          int key=16*key_block+int(cute::get<1>(coordinate));
          int64_t offset=fla_recurrence::state_offset(segment,head,value,key);
          float next=__fadd_rn(result(item,m,n),bias[offset]);
          carries[plane+fla_recurrence::state_offset(
              segment+1,head,value,key)]=next;
        }
    __syncthreads();
    if(segment+2<segment_count) {
      for(int linear=tid;linear<kCarryElements;linear+=kCpPrefixThreads) {
        int value=value_base+linear/128;
        int key=linear%128;
        float next=carries[plane+fla_recurrence::state_offset(
            segment+1,head,value,key)];
        stage.carry[linear]=fla_recurrence::round_bf16(next);
      }
      __syncthreads();
    }
  }
#else
  __trap();
#endif
}
}  // namespace

extern "C" hggcError_t configure_fla_cp_pretriangular_output() {
  auto configure=[](auto kernel) {
    return hggcFuncSetAttribute(kernel,
        hggcFuncAttributeMaxDynamicSharedMemorySize,
        FLA_RECURRENCE_DYNAMIC_SHARED_BYTES);
  };
#if !defined(FLA_CP_BV16_VARIANT)
  if constexpr(FLA_HEADS<=8) {
    auto small_status=FLA_JOIN_HEAD(configure_fla_cp_pretriangular_output_bv16, FLA_HEADS)();
    if(small_status!=hggcSuccess) return small_status;
  }
  auto status=configure(fla_cp_pretriangular_output<fla_recurrence::CpPass::LegacyZero>);
  if(status!=hggcSuccess) return status;
  status=configure(fla_cp_pretriangular_output<fla_recurrence::CpPass::LegacyRepair>);
  if(status!=hggcSuccess) return status;
  status=configure(fla_cp_pretriangular_output<fla_recurrence::CpPass::Zero>);
#else
  auto status=configure(fla_cp_pretriangular_output<fla_recurrence::CpPass::Zero>);
#endif
  if(status!=hggcSuccess) return status;
  status=configure(fla_cp_pretriangular_output<fla_recurrence::CpPass::Transition>);
  if(status!=hggcSuccess) return status;
  return configure(fla_cp_pretriangular_output<fla_recurrence::CpPass::Correction>);
}

// repair_segment=-2 selects the fixed state-passing graph.  -1 launches the
// legacy independent pre-scan, and a positive value launches one legacy repair
// segment.  The selector is internal to the versioned CP ABI.
extern "C" hggcError_t launch_fla_cp_pretriangular_output(
    void const* qg,void const* a,void* output,void const* kg,void const* u,
    void const* w,float const* gate,void const* initial_state,
    void* final_state,int state_is_fp32,int32_t const* state_indices,
    float* ends,float* checkpoints,int* executed,int32_t const* bounds,
    int64_t const* offsets,int count,int repair_segment,hggcStream_t stream) {
  if(count<1 || count>8 || repair_segment < -2 || repair_segment==0 ||
      repair_segment>=count || state_is_fp32<0 || state_is_fp32>1)
    return hggcErrorInvalidValue;
  if(!qg || !a || !output || !kg || !u || !w || !gate || !ends ||
      !checkpoints || !executed || !bounds || !offsets)
    return hggcErrorInvalidValue;
#if !defined(FLA_CP_BV16_VARIANT)
  if(repair_segment==-2 && fla_cp_value_slice::use_bv16(FLA_HEADS,count))
    return FLA_JOIN_HEAD(launch_fla_cp_pretriangular_output_bv16, FLA_HEADS)(
        qg,a,output,kg,u,w,gate,initial_state,final_state,state_is_fp32,
        state_indices,ends,checkpoints,executed,bounds,offsets,count,
        repair_segment,stream);
#else
  // This private entry point must never serve the legacy graph.
  if(repair_segment!=-2) return hggcErrorInvalidValue;
#endif
  auto launch_pass=[&](auto pass,int grid_segments,int selected_segment) {
    constexpr auto kPass=decltype(pass)::value;
    fla_cp_pretriangular_output<kPass><<<dim3(128/fla_recurrence::kBv,FLA_HEADS,grid_segments),128,
        FLA_RECURRENCE_DYNAMIC_SHARED_BYTES,stream>>>(
        static_cast<uint16_t const*>(qg),static_cast<uint16_t const*>(a),
        static_cast<uint16_t*>(output),static_cast<uint16_t const*>(kg),
        static_cast<uint16_t const*>(u),static_cast<uint16_t const*>(w),gate,
        initial_state,final_state,state_is_fp32,state_indices,ends,checkpoints,
        executed,bounds,offsets,selected_segment,count);
    return hggcGetLastError();
  };
#if !defined(FLA_CP_BV16_VARIANT)
  using LegacyZero=std::integral_constant<fla_recurrence::CpPass,
                                          fla_recurrence::CpPass::LegacyZero>;
  using LegacyRepair=std::integral_constant<fla_recurrence::CpPass,
                                            fla_recurrence::CpPass::LegacyRepair>;
#endif
  using Zero=std::integral_constant<fla_recurrence::CpPass,
                                    fla_recurrence::CpPass::Zero>;
  using Transition=std::integral_constant<fla_recurrence::CpPass,
                                          fla_recurrence::CpPass::Transition>;
  using Correction=std::integral_constant<fla_recurrence::CpPass,
                                          fla_recurrence::CpPass::Correction>;
#if !defined(FLA_CP_BV16_VARIANT)
  if(repair_segment==-1)
    return launch_pass(LegacyZero{},count,-1);
  if(repair_segment>0)
    return launch_pass(LegacyRepair{},1,repair_segment);
#endif

  auto status=launch_pass(Zero{},count,-1);
  if(status!=hggcSuccess || count==1) return status;
  // Segment zero already produced C[1]; M[S-1] has no successor carry
  // consumer. Launch only M[1..S-2], with an offset in the kernel grid.
  // This host condition depends only on the fixed segment plan, not inputs.
  if(count>2) {
    status=launch_pass(Transition{},count-2,-1);
    if(status!=hggcSuccess) return status;
  }
  fla_cp_state_prefix<FLA_HEADS><<<
      dim3(128/kCpPrefixRows,FLA_HEADS),kCpPrefixThreads,0,stream>>>(
      ends,checkpoints,checkpoints,initial_state,final_state,state_is_fp32,
      state_indices,count);
  status=hggcGetLastError();
  if(status!=hggcSuccess) return status;
  return launch_pass(Correction{},count-1,-1);
}
