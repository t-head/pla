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
// Private frozen-domain GLM-5.3 H8 FLA recurrence candidate for tc01.
//
// Source/CPU contract only: this file is intentionally not wired into the
// production dispatcher.  Fixed-image compilation, exact emitted resources,
// fragment/basis correctness, full numerical parity, and performance are all
// later fail-closed gates.
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
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/arch/mma_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>

#ifndef FLA_RECURRENCE_AIU_W
#define FLA_RECURRENCE_AIU_W 0
#endif
#ifndef FLA_RECURRENCE_AIU_K
#define FLA_RECURRENCE_AIU_K 0
#endif
#ifndef FLA_RECURRENCE_VEC_IO
#define FLA_RECURRENCE_VEC_IO 0
#endif
#ifndef FLA_RECURRENCE_AIU_V
#define FLA_RECURRENCE_AIU_V 0
#endif
#ifndef FLA_RECURRENCE_SHFL_DECAY
#define FLA_RECURRENCE_SHFL_DECAY 0
#endif
#ifndef FLA_RECURRENCE_CPASYNC_W
#define FLA_RECURRENCE_CPASYNC_W 0
#endif
#ifndef FLA_RECURRENCE_OVERLAP_AIU
#define FLA_RECURRENCE_OVERLAP_AIU 0
#endif
#ifndef FLA_RECURRENCE_REUSE_K_B
#define FLA_RECURRENCE_REUSE_K_B 0
#endif
#ifndef FLA_RECURRENCE_SEPARATE_V_STAGE
#define FLA_RECURRENCE_SEPARATE_V_STAGE 0
#endif
#ifndef FLA_RECURRENCE_FRAGMENT_RESIDUAL
#define FLA_RECURRENCE_FRAGMENT_RESIDUAL 0
#endif
#ifndef FLA_RECURRENCE_SHFL_HISTORY
#define FLA_RECURRENCE_SHFL_HISTORY 0
#endif
#ifndef FLA_RECURRENCE_DIRECT_UPDATE
#define FLA_RECURRENCE_DIRECT_UPDATE 0
#endif
#ifndef FLA_RECURRENCE_LOCAL_DECAY
#define FLA_RECURRENCE_LOCAL_DECAY 0
#endif
#ifndef FLA_RECURRENCE_REFERENCE_EXP
#define FLA_RECURRENCE_REFERENCE_EXP 0
#endif
#ifndef FLA_RECURRENCE_CPASYNC_K
#define FLA_RECURRENCE_CPASYNC_K 0
#endif
#ifndef FLA_RECURRENCE_PIPELINE5
#define FLA_RECURRENCE_PIPELINE5 0
#endif
#ifndef FLA_RECURRENCE_PIPELINE5_AIU
#define FLA_RECURRENCE_PIPELINE5_AIU 0
#endif
#ifndef FLA_RECURRENCE_PIPELINE5_AIU_LEADER_WAIT
#define FLA_RECURRENCE_PIPELINE5_AIU_LEADER_WAIT 0
#endif
#ifndef FLA_RECURRENCE_SCRATCH_HISTORY
#define FLA_RECURRENCE_SCRATCH_HISTORY 0
#endif
#ifndef FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
#define FLA_RECURRENCE_REFERENCE_STATE_LAYOUT 0
#endif
#ifndef FLA_RECURRENCE_FULL_ROWS_SPECIALIZATION
#define FLA_RECURRENCE_FULL_ROWS_SPECIALIZATION 0
#endif
#ifndef FLA_RECURRENCE_PIPELINE5_COMPACT_CPASYNC
#define FLA_RECURRENCE_PIPELINE5_COMPACT_CPASYNC 0
#endif
#ifndef FLA_RECURRENCE_BV16
#define FLA_RECURRENCE_BV16 0
#endif
#ifndef FLA_RECURRENCE_PRECOMPUTE_DECAY
#define FLA_RECURRENCE_PRECOMPUTE_DECAY 0
#endif
#ifndef FLA_RECURRENCE_EXACT_FRAGMENT_COORDS
#define FLA_RECURRENCE_EXACT_FRAGMENT_COORDS 0
#endif
#ifndef FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL
#define FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL 0
#endif
#ifndef FLA_RECURRENCE_DIRECT_V_LOAD
#define FLA_RECURRENCE_DIRECT_V_LOAD 0
#endif
#ifndef FLA_RECURRENCE_PACKED_RESIDUAL_CONVERT
#define FLA_RECURRENCE_PACKED_RESIDUAL_CONVERT 0
#endif
#ifndef FLA_RECURRENCE_DIRECT_HISTORY_SHFL
#define FLA_RECURRENCE_DIRECT_HISTORY_SHFL 0
#endif
#ifndef FLA_RECURRENCE_SKIP_BFRAGMENT_BARRIER
#define FLA_RECURRENCE_SKIP_BFRAGMENT_BARRIER 0
#endif
#ifndef FLA_RECURRENCE_SKIP_LONG_BFRAGMENT_BARRIER
#define FLA_RECURRENCE_SKIP_LONG_BFRAGMENT_BARRIER 0
#endif
#ifndef FLA_RECURRENCE_BFRAGMENT_HISTORY
#define FLA_RECURRENCE_BFRAGMENT_HISTORY 0
#endif
#ifndef FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
#define FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY 0
#endif
#ifndef FLA_RECURRENCE_MIN_BLOCKS_PER_CU
#define FLA_RECURRENCE_MIN_BLOCKS_PER_CU 1
#endif
#ifndef FLA_RECURRENCE_BODY_ONLY
#define FLA_RECURRENCE_BODY_ONLY 0
#endif
#if FLA_RECURRENCE_SCRATCH_HISTORY
#define FLA_RECURRENCE_DYNAMIC_SHARED_BYTES 49152
#else
#define FLA_RECURRENCE_DYNAMIC_SHARED_BYTES 45056
#endif

namespace cute {

// This reader shape is an implementation hypothesis for the first compile
// candidate.  The 4-warp 64x32 recurrence fragment mapping must be proven by
// an isolated basis/device gate; the 8-warp recompute proof is not inherited.
template<bool Trans> struct FlaRecurrenceRawReader {};

template<bool Trans>
struct Copy_Traits<FlaRecurrenceRawReader<Trans>> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32,_128>,Stride<_128,_1>>;
  using DstLayout = Layout<Shape<_32,Shape<_32,_4>>,
                           Stride<_32,Stride<_1,_1024>>>;
  using RefLayout = DstLayout;

  template<class TS,class SL,class TD,class DL>
  CUTE_HOST_DEVICE friend void copy_unpack(
      Copy_Traits const&, Tensor<TS,SL> const& src, Tensor<TD,DL>& dst) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    static_assert(is_mix_iterator<typename TS::iterator>::value);
    auto coord=src.data().coord_;
    int row=Trans ? int(get<0>(coord)) : int(get<1>(coord));
    int col=Trans ? int(get<1>(coord)) : int(get<0>(coord));
    uint32_t base=static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(src.data().ptr_.get()));
    uint32_t address=base+2*(64*row+(col^(8*(row%8))));
    auto* words=reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(dst.data()));
    if constexpr (Trans) {
      asm volatile(
          "ppu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 "
          "{%0, %1, %2, %3}, [%4];"
          : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3])
          : "r"(address));
    } else {
      asm volatile(
          "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
          "{%0, %1, %2, %3}, [%4];"
          : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3])
          : "r"(address));
    }
#else
    CUTE_INVALID_CONTROL_PATH("FLA recurrence requires the tc01 device image");
#endif
  }
};

}  // namespace cute

namespace fla_recurrence {

#if FLA_RECURRENCE_BV16
constexpr int kBv = 16;
#else
constexpr int kBv = 32;
#endif

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
using namespace cute;

constexpr int kThreads = 128;
constexpr int kBt = 64;
constexpr int kStateOrdinals = kBv/2;
#if FLA_HEADS > 0
constexpr int kHeads = FLA_HEADS;
#else
#define kHeads FLA_DEVICE_HEADS(y)
#endif
constexpr int kDim = 128;
#if FLA_RECURRENCE_SCRATCH_HISTORY
constexpr int kDynamicSharedBytes = 49152;
#else
constexpr int kDynamicSharedBytes = 45056;
#endif

using Mma=decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{},
    Layout<Shape<_4,_1,_1>,Stride<_1,_4,_4>>{}));
using AReader=Copy_Atom<FlaRecurrenceRawReader<false>,bfloat16_t>;
using ATransReader=Copy_Atom<FlaRecurrenceRawReader<true>,bfloat16_t>;
using BReader=Copy_Atom<FlaRecurrenceRawReader<true>,bfloat16_t>;
using AiuAReader=Copy_Atom<PPU0010_TSM_LD_SWZL<
    bfloat16_t,64,64,/*Swap=*/false,/*Trans=*/false,/*InstNum=*/1>,bfloat16_t>;
using AiuATransReader=Copy_Atom<PPU0010_TSM_LD_SWZL<
    bfloat16_t,64,64,/*Swap=*/false,/*Trans=*/true,/*InstNum=*/1>,bfloat16_t>;
using ALogical=Layout<Shape<_64,_64>,Stride<_64,_1>>;
using KTokenMajorLogical=Layout<Shape<_64,_64>,Stride<_1,_64>>;
// CuTe partitions B as logical (N,K); storage remains row-major [K,N].
using BLogical=Layout<Shape<Int<kBv>,_64>,Stride<_1,_64>>;

struct alignas(128) Stage {
  uint16_t a[64*64];
  uint16_t b[64*64];  // second half is deliberate padding for the raw reader.
  float update[64*32];
#if FLA_RECURRENCE_SEPARATE_V_STAGE
  uint16_t v[64*kBv];
#endif
#if FLA_RECURRENCE_PIPELINE5
  // The pipeline addresses the full allocation through fixed byte offsets.
  // Pad the C++ object to the launch-time 44 KiB so those accesses stay within
  // the declared object rather than relying on dynamic-smem tail storage.
  uint16_t pipeline_tail[10240];
#if FLA_RECURRENCE_SCRATCH_HISTORY
  uint16_t history_tail[2048];
#endif
#endif
};
static_assert(sizeof(Stage) <= kDynamicSharedBytes);
#if FLA_RECURRENCE_PIPELINE5
static_assert(sizeof(Stage) == kDynamicSharedBytes);
#endif

union Bf16Bits {
  uint16_t bits;
  __ppu_bfloat16 value;
};

union Bf16Pack8 {
  uint4 vector;
  uint32_t pair[4];
  uint16_t element[8];
};

CUTE_DEVICE uint16_t round_bf16(float value) {
  Bf16Bits result;
  result.value=__float2bfloat16_rn(value);
  return result.bits;
}

CUTE_DEVICE float load_bf16(uint16_t bits) {
  Bf16Bits value;
  value.bits=bits;
  return __bfloat162float(value.value);
}

CUTE_DEVICE float split_exp(float x) {
  float y=__fmul_rn(x,__uint_as_float(0x3fb8aa3bu));
  bool low=y < -126.0f;
  float core=__exp2f(__fmul_rn(y,low ? 0.5f : 1.0f));
  return __fmul_rn(core,low ? core : 1.0f);
}

// The frozen FLA-Triton tc01 image lowers the recurrence decay to one
// branch-free ex2.approx after multiplication by log2(e).  Keep the guarded
// underflow split above as the fallback; this exact lowering is isolated until
// device correctness establishes that the GLM-5.3 domain never needs it.
CUTE_DEVICE float reference_exp(float x) {
  return __exp2f(__fmul_rn(x,__uint_as_float(0x3fb8aa3bu)));
}

CUTE_DEVICE float recurrence_exp(float x) {
#if FLA_RECURRENCE_REFERENCE_EXP
  return reference_exp(x);
#else
  return split_exp(x);
#endif
}

FLA_HEAD_OFFSET int64_t token_offset(
    int64_t token,int head,int feature,int width=kDim);

CUTE_DEVICE void load_decay_pair_shfl(
    float const* gk,int64_t last,int head,int panel,float& decay0,float& decay1) {
  int lane=threadIdx.x%32;
  int key_base=(lane>>2)+16*(threadIdx.x/32);
  decay0=0.0f;
  decay1=0.0f;
  if((lane&3)==0) {
    decay0=split_exp(gk[token_offset(last,head,64*panel+key_base)]);
    decay1=split_exp(gk[token_offset(last,head,64*panel+key_base+8)]);
  }
  decay0=__shfl_sync(0xffffffffu,decay0,lane&~3,32);
  decay1=__shfl_sync(0xffffffffu,decay1,lane&~3,32);
}

CUTE_HOST_DEVICE int recurrence_state_v(int tid,int ordinal) {
  int lane=tid%32;
  int b0=ordinal&1;
  int b1=(ordinal>>1)&1;
  int b3=(ordinal>>3)&1;
  return (lane & 3) + 4*b0 + 8*b1 + 16*b3;
}

CUTE_HOST_DEVICE int recurrence_state_k(int tid,int ordinal) {
  int lane=tid%32,warp=tid/32;
  int b2=(ordinal>>2)&1;
  return (lane >> 2) + 8*b2 + 16*warp;
}

CUTE_DEVICE void load_decay_pair_local(
    float const* gk,int64_t last,int head,int panel,float& decay0,float& decay1) {
  int key0=recurrence_state_k(threadIdx.x,0);
  int key1=recurrence_state_k(threadIdx.x,4);
  decay0=recurrence_exp(gk[token_offset(last,head,64*panel+key0)]);
  decay1=recurrence_exp(gk[token_offset(last,head,64*panel+key1)]);
}

FLA_HEAD_OFFSET int64_t token_offset(
    int64_t token,int head,int feature,int width) {
  return (token*int64_t(kHeads)+head)*int64_t(width)+feature;
}

FLA_HEAD_OFFSET int64_t history_offset(
    int64_t chunk,int head,int value,int key) {
  return ((chunk*int64_t(kHeads)+head)*kDim+value)*kDim+key;
}

FLA_HEAD_OFFSET int64_t state_offset(
    int64_t slot,int head,int value,int key) {
  return ((slot*int64_t(kHeads)+head)*kDim+value)*kDim+key;
}

CUTE_DEVICE int swizzled(int row,int col) {
  return 64*row+(col^(8*(row%8)));
}

FLA_HEAD_OFFSET int64_t output_native_history_offset(
    int64_t chunk,int head,int value_base,int panel,int local) {
  int value_tile=value_base/64;
  int value_quarter=(value_base%64)/16;
  int tile=2*value_tile+panel;
  return ((((chunk*int64_t(kHeads)+head)*4+tile)*4096)
          +value_quarter*1024+local);
}

template<class Pointer>
CUTE_DEVICE auto b_source_slice(Pointer pointer,int tid,int fragment,int q) {
  int lane=tid%32;
  auto iterator=make_ptr_inttuple_mix_iter(
      make_coord(16*q+lane%8+8*((lane>>4)&1),
                 16*fragment+(lane&8)),pointer);
  return make_tensor(iterator,make_layout(make_shape(_8{})));
}

template<class ACopyAtom,class Accumulator>
CUTE_DEVICE void mma_accumulate(Stage& stage,Mma const& mma,
                                Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(stage.a))));
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(stage.b))));
  auto a=make_tensor(ap,ALogical{});
  auto b=make_tensor(bp,BLogical{});
  auto thr=mma.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  auto r_b=thr.partition_fragment_B(b);
  auto copy_a=make_tiled_copy_A(ACopyAtom{},mma);
  auto copy_b=make_tiled_copy_B(BReader{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  auto d_b=copy_b.get_thread_slice(tid).retile_D(r_b);
  CUTE_STATIC_ASSERT_V(size<2>(r_a)==Int<4>{});
  CUTE_STATIC_ASSERT_V(size<1>(r_b)==Int<kBv/16>{});
  CUTE_STATIC_ASSERT_V(size<2>(r_b)==Int<4>{});
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    CUTE_UNROLL
    for(int fragment=0;fragment<kBv/16;++fragment) {
      auto src=b_source_slice(bp,tid,fragment,q);
      auto dst=d_b(_,fragment,q);
      BReader{}.call(src,dst);
    }
    cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
  }
}

template<class BFragment>
CUTE_DEVICE void load_b_fragment(
    Stage& stage,Mma const& mma,BFragment& fragment) {
  int tid=threadIdx.x;
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(stage.b))));
  auto b=make_tensor(bp,BLogical{});
  auto copy_b=make_tiled_copy_B(BReader{},mma);
  auto d_b=copy_b.get_thread_slice(tid).retile_D(fragment);
  CUTE_STATIC_ASSERT_V(size<1>(fragment)==Int<kBv/16>{});
  CUTE_STATIC_ASSERT_V(size<2>(fragment)==Int<4>{});
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    CUTE_UNROLL
    for(int part=0;part<kBv/16;++part) {
      auto src=b_source_slice(bp,tid,part,q);
      auto dst=d_b(_,part,q);
      BReader{}.call(src,dst);
    }
  }
}

template<class ACopyAtom,class BFragment,class Accumulator>
CUTE_DEVICE void mma_accumulate_with_b(
    Stage& stage,Mma const& mma,BFragment const& b_fragment,
    Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(stage.a))));
  auto a=make_tensor(ap,ALogical{});
  auto thr=mma.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  auto copy_a=make_tiled_copy_A(ACopyAtom{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  CUTE_STATIC_ASSERT_V(size<2>(r_a)==Int<4>{});
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    cute::gemm(mma,r_a(_,_,q),b_fragment(_,_,q),accumulator);
  }
}

template<class BFragment,class Accumulator>
CUTE_DEVICE void mma_accumulate_k_token_major_with_b(
    Stage& stage,Mma const& mma,BFragment const& b_fragment,
    Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(stage.a))));
  auto a=make_tensor(ap,ALogical{});
  auto thr=mma.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  CUTE_STATIC_ASSERT_V(size<2>(r_a)==Int<4>{});
  int address0=(((tid&7)<<6)|((tid&7)<<3))^
               (((tid&96)>>1)|(tid&8));
  address0|=(tid&16)<<5;
  uint32_t shared_base=static_cast<uint32_t>(
      __cvta_generic_to_shared(stage.a));
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    auto fragment=r_a(_,_,q);
    auto* words=reinterpret_cast<uint32_t*>(
        cute::raw_pointer_cast(fragment.data()));
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    uint32_t address=shared_base+2*address0+2048*q;
    asm volatile(
        "ppu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3])
        : "r"(address));
#endif
    cute::gemm(mma,r_a(_,_,q),b_fragment(_,_,q),accumulator);
  }
}

CUTE_DEVICE void store_history_and_publish_state_scalar(
    Stage& stage,uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  int tid=threadIdx.x;
  CUTE_UNROLL
  for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
    int value=recurrence_state_v(tid,ordinal);
    int key=recurrence_state_k(tid,ordinal);
    uint16_t rounded=round_bf16(carried[panel][ordinal]);
    h[history_offset(history_chunk,head,value_base+value,64*panel+key)]=rounded;
    stage.b[swizzled(key,value)]=rounded;
  }
}

CUTE_DEVICE void store_history_and_publish_state_vec(
    Stage& stage,uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  int tid=threadIdx.x;
  auto* scratch=reinterpret_cast<uint16_t*>(stage.update);
  CUTE_UNROLL
  for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
    int value=recurrence_state_v(tid,ordinal);
    int key=recurrence_state_k(tid,ordinal);
    uint16_t rounded=round_bf16(carried[panel][ordinal]);
    stage.b[swizzled(key,value)]=rounded;
    scratch[value*64+key]=rounded;
  }
  __syncthreads();
  for(int linear=8*tid;linear<kBv*64;linear+=8*kThreads) {
    int value=linear/64,key=linear%64;
    auto packed=*reinterpret_cast<uint4 const*>(scratch+linear);
    *reinterpret_cast<uint4*>(h+history_offset(
        history_chunk,head,value_base+value,64*panel+key))=packed;
  }
}

// The recurrence fragment owns a 32x16 state tile per warp.  Transpose that
// tile in registers for the row-major history write: lanes r+4*k own key k
// for value mod 4 == r.  One shuffle carries the low/high eight-key halves,
// so four lanes emit the 16-byte stores without the update-scratch round trip.
CUTE_DEVICE void store_history_and_publish_state_shfl(
    Stage& stage,uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  int tid=threadIdx.x;
  int lane=tid%32;
  int warp=tid/32;
  uint16_t state_bf16[kStateOrdinals];
  CUTE_UNROLL
  for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
    int value=recurrence_state_v(tid,ordinal);
    int key=recurrence_state_k(tid,ordinal);
    uint16_t rounded=round_bf16(carried[panel][ordinal]);
    state_bf16[ordinal]=rounded;
    stage.b[swizzled(key,value)]=rounded;
  }
  CUTE_UNROLL
  for(int value_group=0;value_group<kBv/4;++value_group) {
    int ordinal_low=(value_group&3)+((value_group&4)<<1);
    int ordinal_high=ordinal_low+4;
    unsigned source=unsigned(state_bf16[ordinal_low]) |
                    (unsigned(state_bf16[ordinal_high])<<16);
    Bf16Pack8 low{};
    Bf16Pack8 high{};
    CUTE_UNROLL
    for(int key=0;key<8;++key) {
      int source_lane=(lane&3)+4*key;
      unsigned packed=__shfl_sync(0xffffffffu,source,source_lane,32);
      low.element[key]=uint16_t(packed);
      high.element[key]=uint16_t(packed>>16);
    }
    if(lane<4) {
      int value=4*value_group+lane;
      int64_t base=history_offset(
          history_chunk,head,value_base+value,64*panel+16*warp);
      *reinterpret_cast<uint4*>(h+base)=low.vector;
      *reinterpret_cast<uint4*>(h+base+8)=high.vector;
    }
  }
}

CUTE_DEVICE void store_history_and_publish_state(
    Stage& stage,uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
#if FLA_RECURRENCE_SHFL_HISTORY
  store_history_and_publish_state_shfl(
      stage,h,carried,panel,history_chunk,head,value_base);
#elif FLA_RECURRENCE_VEC_IO
  store_history_and_publish_state_vec(
      stage,h,carried,panel,history_chunk,head,value_base);
#else
  store_history_and_publish_state_scalar(
      stage,h,carried,panel,history_chunk,head,value_base);
#endif
}

CUTE_DEVICE void publish_w_panel_scalar(
    Stage& stage,uint16_t const* w,int64_t first,int rows,int head,int panel) {
  for(int linear=threadIdx.x;linear<64*64;linear+=kThreads) {
    int row=linear/64,key=linear%64;
    stage.a[swizzled(row,key)]=row<rows
        ? w[token_offset(first+row,head,64*panel+key)] : uint16_t(0);
  }
}

// tc01 AIU bulk-copy experiment for the hottest recurrence publisher.  The
// scalar implementation above remains the default/fallback.  W is logically a
// [64 tokens,64 keys] b16 tile with a 8*128-element token pitch.  The swizzled
// AIU destination is the exact XOR layout consumed by FlaRecurrenceRawReader.
CUTE_DEVICE void issue_w_panel_aiu(
    Stage& stage,uint16_t const* w,int64_t first,int rows,int head,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) {
    using AiuLoad=PPU0010_AIU_LOAD<
        C<64*64*16>,bfloat16_t,/*Trans=*/false,/*Swzl=*/true>;
    auto const* source=w+token_offset(first,head,64*panel);
    AiuDesc desc;
    desc.template init<bfloat16_t,/*Trans=*/false,64,64>(
        reinterpret_cast<uint8_t const*>(source),rows,64,
        make_stride(FLA_CUTE_INT(kHeads*kDim),_1{}));
    uint32_t destination=static_cast<uint32_t>(
        __cvta_generic_to_shared(stage.a));
    AiuLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(destination)),
                  reinterpret_cast<void const*>(source),desc,0,0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  }
#else
  publish_w_panel_scalar(stage,w,first,rows,head,panel);
#endif
}

CUTE_DEVICE void wait_panel_aiu() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

CUTE_DEVICE void publish_w_panel_aiu(
    Stage& stage,uint16_t const* w,int64_t first,int rows,int head,int panel) {
  issue_w_panel_aiu(stage,w,first,rows,head,panel);
  wait_panel_aiu();
}

// Match the FLA-Triton tc01 lowering: every thread contributes four aligned
// 16-byte global-to-TSM copies.  The destination uses the same 128-byte XOR
// swizzle consumed by AiuAReader; the fourth cp.async operand zero-fills tail
// rows without a divergent scalar store loop.
CUTE_DEVICE void publish_w_panel_cpasync(
    Stage& stage,uint16_t const* w,int64_t first,int rows,int head,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  CUTE_UNROLL
  for(int segment=threadIdx.x;segment<64*64/8;segment+=kThreads) {
    int row=segment/8;
    int key=8*(segment%8);
    auto const* source=w+token_offset(first+row,head,64*panel+key);
    uint32_t destination=static_cast<uint32_t>(__cvta_generic_to_shared(
        stage.a+swizzled(row,key)));
    int valid_bytes=row<rows ? 16 : 0;
    asm volatile(
        "ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
        :: "r"(destination),"l"(source),"r"(valid_bytes) : "memory");
  }
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  asm volatile("ppu.cp.async.wait_group 0;\n" ::: "memory");
#else
  publish_w_panel_scalar(stage,w,first,rows,head,panel);
#endif
}

CUTE_DEVICE void publish_w_panel(
    Stage& stage,uint16_t const* w,int64_t first,int rows,int head,int panel) {
#if FLA_RECURRENCE_CPASYNC_W
  publish_w_panel_cpasync(stage,w,first,rows,head,panel);
#elif FLA_RECURRENCE_AIU_W
  publish_w_panel_aiu(stage,w,first,rows,head,panel);
#else
  publish_w_panel_scalar(stage,w,first,rows,head,panel);
#endif
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void publish_residual_scalar(
    Stage& stage,uint16_t const* v,uint16_t* v_new,
    Accumulator const& residual,Coordinates const& coordinates,
    int64_t first,int rows,int head,int value_base) {
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(residual));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(residual));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(residual));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int row=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        uint16_t rounded=uint16_t(0);
        if(row<rows && value<kBv) {
          float input=load_bf16(v[token_offset(first+row,head,value_base+value)]);
          rounded=round_bf16(__fsub_rn(input,residual(value_index,m,n)));
          v_new[token_offset(first+row,head,value_base+value)]=rounded;
        }
        if(value<kBv) stage.b[swizzled(row,value)]=rounded;
      }
}

CUTE_DEVICE void issue_v_panel_aiu(
    Stage& stage,uint16_t const* v,int64_t first,int rows,int head,int value_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) {
    using AiuLoad=PPU0010_AIU_LOAD<
        C<64*kBv*16>,bfloat16_t,/*Trans=*/false,/*Swzl=*/false>;
    auto const* source=v+token_offset(first,head,value_base);
    AiuDesc desc;
    desc.template init<bfloat16_t,/*Trans=*/false,64,kBv>(
        reinterpret_cast<uint8_t const*>(source),rows,kBv,
        make_stride(FLA_CUTE_INT(kHeads*kDim),_1{}));
    uint32_t destination=static_cast<uint32_t>(
#if FLA_RECURRENCE_SEPARATE_V_STAGE
        __cvta_generic_to_shared(stage.v));
#else
        __cvta_generic_to_shared(stage.a));
#endif
    AiuLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(destination)),
                  reinterpret_cast<void const*>(source),desc,0,0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  }
#else
  (void)stage;(void)v;(void)first;(void)rows;(void)head;(void)value_base;
#endif
}

CUTE_DEVICE void wait_v_panel_aiu() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void publish_residual_vec(
    Stage& stage,uint16_t const* v,uint16_t* v_new,
    Accumulator const& residual,Coordinates const& coordinates,
    int64_t first,int rows,int head,int value_base) {
#if FLA_RECURRENCE_AIU_V
  issue_v_panel_aiu(stage,v,first,rows,head,value_base);
#endif
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(residual));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(residual));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(residual));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int row=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        if(row<64 && value<kBv)
          stage.update[row*kBv+value]=residual(value_index,m,n);
      }
#if FLA_RECURRENCE_AIU_V
  wait_v_panel_aiu();
#endif
  __syncthreads();
  for(int linear=8*threadIdx.x;linear<64*kBv;linear+=8*kThreads) {
    int row=linear/kBv,value=linear%kBv;
    Bf16Pack8 output{};
    if(row<rows) {
#if FLA_RECURRENCE_AIU_V
#if FLA_RECURRENCE_SEPARATE_V_STAGE
      auto input=*reinterpret_cast<Bf16Pack8 const*>(
          stage.v+row*kBv+value);
#else
      auto input=*reinterpret_cast<Bf16Pack8 const*>(stage.a+row*kBv+value);
#endif
#else
      auto input=*reinterpret_cast<Bf16Pack8 const*>(
          v+token_offset(first+row,head,value_base+value));
#endif
      CUTE_UNROLL
      for(int i=0;i<8;++i) {
        output.element[i]=round_bf16(__fsub_rn(
            load_bf16(input.element[i]),stage.update[row*kBv+value+i]));
      }
      *reinterpret_cast<uint4*>(
          v_new+token_offset(first+row,head,value_base+value))=output.vector;
    }
    *reinterpret_cast<uint4*>(stage.b+swizzled(row,value))=output.vector;
  }
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void publish_residual_fragment(
    Stage& stage,uint16_t const* v,uint16_t* v_new,
    Accumulator const& residual,Coordinates const& coordinates,
    int64_t first,int rows,int head,int value_base) {
#if FLA_RECURRENCE_AIU_V && FLA_RECURRENCE_SEPARATE_V_STAGE
  issue_v_panel_aiu(stage,v,first,rows,head,value_base);
  wait_v_panel_aiu();
  __syncthreads();
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(residual));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(residual));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(residual));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int row=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        uint16_t rounded=uint16_t(0);
        if(row<rows && value<kBv) {
          float input=load_bf16(stage.v[row*kBv+value]);
          rounded=round_bf16(__fsub_rn(
              input,residual(value_index,m,n)));
          v_new[token_offset(first+row,head,value_base+value)]=rounded;
        }
        if(value<kBv) stage.b[swizzled(row,value)]=rounded;
      }
#else
  publish_residual_scalar(
      stage,v,v_new,residual,coordinates,first,rows,head,value_base);
#endif
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void publish_residual(
    Stage& stage,uint16_t const* v,uint16_t* v_new,
    Accumulator const& residual,Coordinates const& coordinates,
    int64_t first,int rows,int head,int value_base) {
#if FLA_RECURRENCE_FRAGMENT_RESIDUAL
  publish_residual_fragment(
      stage,v,v_new,residual,coordinates,first,rows,head,value_base);
#elif FLA_RECURRENCE_VEC_IO
  publish_residual_vec(
      stage,v,v_new,residual,coordinates,first,rows,head,value_base);
#else
  publish_residual_scalar(
      stage,v,v_new,residual,coordinates,first,rows,head,value_base);
#endif
}

CUTE_DEVICE void publish_kg_panel_scalar(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
  for(int linear=threadIdx.x;linear<64*64;linear+=kThreads) {
    int key=linear/64,row=linear%64;
    stage.a[swizzled(key,row)]=row<rows
        ? k[token_offset(first+row,head,64*panel+key)] : uint16_t(0);
  }
}

template<int SegmentOffset>
CUTE_DEVICE void issue_kg_segment_cpasync(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int segment=threadIdx.x+SegmentOffset;
  int row=segment/8;
  int key=8*(segment%8);
  auto const* source=k+token_offset(first+row,head,64*panel+key);
  uint32_t destination=static_cast<uint32_t>(__cvta_generic_to_shared(
      stage.a+swizzled(row,key)));
  int valid_bytes=row<rows ? 16 : 0;
  asm volatile(
      "ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid_bytes) : "memory");
#else
  (void)stage;(void)k;(void)first;(void)rows;(void)head;(void)panel;
#endif
}

// FLA-Triton stores K token-major into the same 128-byte XOR-swizzled byte
// image as W, then transposes while loading the MMA A fragment.  Each tc01
// thread owns four compile-time-unrolled aligned 16-byte copies.
CUTE_DEVICE void issue_kg_panel_cpasync(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  issue_kg_segment_cpasync<0>(stage,k,first,rows,head,panel);
  issue_kg_segment_cpasync<128>(stage,k,first,rows,head,panel);
  issue_kg_segment_cpasync<256>(stage,k,first,rows,head,panel);
  issue_kg_segment_cpasync<384>(stage,k,first,rows,head,panel);
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#else
  publish_kg_panel_scalar(stage,k,first,rows,head,panel);
#endif
}

CUTE_DEVICE void wait_kg_panel_cpasync() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.wait_group 0;\n" ::: "memory");
#endif
}

CUTE_DEVICE void issue_kg_panel_aiu(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) {
    using AiuLoad=PPU0010_AIU_LOAD<
        C<64*64*16>,bfloat16_t,/*Trans=*/true,/*Swzl=*/true>;
    auto const* source=k+token_offset(first,head,64*panel);
    AiuDesc desc;
    desc.template init<bfloat16_t,/*Trans=*/true,64,64>(
        reinterpret_cast<uint8_t const*>(source),64,rows,
        make_stride(_1{},FLA_CUTE_INT(kHeads*kDim)));
    uint32_t destination=static_cast<uint32_t>(
        __cvta_generic_to_shared(stage.a));
    AiuLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(destination)),
                  reinterpret_cast<void const*>(source),desc,0,0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  }
#else
  publish_kg_panel_scalar(stage,k,first,rows,head,panel);
#endif
}

CUTE_DEVICE void publish_kg_panel_aiu(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
  issue_kg_panel_aiu(stage,k,first,rows,head,panel);
  wait_panel_aiu();
}

CUTE_DEVICE void publish_kg_panel(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
#if FLA_RECURRENCE_CPASYNC_K
  issue_kg_panel_cpasync(stage,k,first,rows,head,panel);
  wait_kg_panel_cpasync();
#elif FLA_RECURRENCE_AIU_K
  publish_kg_panel_aiu(stage,k,first,rows,head,panel);
#else
  publish_kg_panel_scalar(stage,k,first,rows,head,panel);
#endif
}

CUTE_DEVICE void compute_update_panel(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head,int panel) {
  Mma mma;
  auto thr=mma.get_thread_slice(threadIdx.x);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},Int<kBv>{})));
  auto update=Mma::make_fragment_C(coordinates);
  CUTE_STATIC_ASSERT_V(size<0>(update)==Int<8>{});
  CUTE_STATIC_ASSERT_V(size<1>(update)==Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(update)==Int<kBv/16>{});
  publish_kg_panel(stage,k,first,rows,head,panel);
  __syncthreads();
  clear(update);
#if FLA_RECURRENCE_CPASYNC_K
  mma_accumulate<AiuATransReader>(stage,mma,update);
#elif FLA_RECURRENCE_AIU_K
  mma_accumulate<AiuATransReader>(stage,mma,update);
#else
  mma_accumulate<AReader>(stage,mma,update);
#endif
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(update));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(update));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(update));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int key=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        if(key<64 && value<kBv)
          stage.update[key*kBv+value]=update(value_index,m,n);
      }
  __syncthreads();
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void compute_update_panel_loaded(
    Stage& stage,Mma const& mma,Accumulator& update,
    Coordinates const& coordinates) {
  (void)coordinates;
  clear(update);
  mma_accumulate<AiuATransReader>(stage,mma,update);
}

template<class Accumulator,class Coordinates,class BFragment>
CUTE_DEVICE void compute_update_panel_loaded(
    Stage& stage,Mma const& mma,Accumulator& update,
    Coordinates const& coordinates,BFragment const& b_fragment) {
  (void)coordinates;
  clear(update);
  mma_accumulate_with_b<AiuATransReader>(
      stage,mma,b_fragment,update);
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void store_update_panel(
    Stage& stage,Accumulator const& update,Coordinates const& coordinates) {
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(update));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(update));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(update));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int key=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        if(key<64 && value<kBv)
          stage.update[key*kBv+value]=update(value_index,m,n);
      }
}

// PPU0010 CLayout maps the eight atom values and the two N tiles to the
// same (key,value) owner as recurrence_state_{k,v}(tid,ordinal).  Keep K^T V in
// that owner's accumulator instead of exchanging it through stage.update.
template<bool UseDecayCache,class Accumulator>
CUTE_DEVICE void update_carried_from_fragment(
    float (&carried)[2][kStateOrdinals],int panel,Accumulator const& update,
    float const* gk,float const* decay_cache,int chunk,int64_t last,int head) {
#if FLA_RECURRENCE_SHFL_DECAY
  float decay0,decay1;
  if constexpr (!UseDecayCache)
    load_decay_pair_shfl(gk,last,head,panel,decay0,decay1);
#elif FLA_RECURRENCE_LOCAL_DECAY
  float decay0,decay1;
  if constexpr (!UseDecayCache)
    load_decay_pair_local(gk,last,head,panel,decay0,decay1);
#endif
  CUTE_UNROLL
  for(int n=0;n<kBv/16;++n) {
    CUTE_UNROLL
    for(int value_index=0;value_index<8;++value_index) {
      int ordinal=value_index+8*n;
      int key=recurrence_state_k(threadIdx.x,ordinal);
      float dot=update(value_index,0,n);
      float decay;
      if constexpr (UseDecayCache) {
        decay=decay_cache[(int64_t(chunk)*kHeads+head)*kDim+64*panel+key];
      } else {
#if FLA_RECURRENCE_SHFL_DECAY || FLA_RECURRENCE_LOCAL_DECAY
        decay=((ordinal>>2)&1) ? decay1 : decay0;
#else
        float g_last=gk[token_offset(last,head,64*panel+key)];
        decay=split_exp(g_last);
#endif
      }
      carried[panel][ordinal]=
          __fmaf_rn(carried[panel][ordinal],decay,dot);
    }
  }
}

#if FLA_RECURRENCE_PIPELINE5
constexpr int kPipeW0=0;
constexpr int kPipeW1=4096;
constexpr int kPipeK0=8192;
constexpr int kPipeK1=12288;
constexpr int kPipeV=16384;
constexpr int kPipeScratch=18432;
constexpr int kPipeHistoryScratch=22528;
#define kPipeTokenStride (kHeads*kDim)

CUTE_DEVICE void issue_pipe_tile16_compact(
    uint16_t* destination_base,uint16_t const* source_base,
    int64_t first,int rows,int head,int feature_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int tid=threadIdx.x;
  int row=tid>>3;
  int feature=8*(tid&7);
  auto const* source=source_base+
      token_offset(first+row,head,feature_base+feature);
  uint32_t destination=static_cast<uint32_t>(__cvta_generic_to_shared(
      destination_base+swizzled(row,feature)));
  int valid=row<rows ? 16 : 0;
  asm volatile("ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  source+=kPipeTokenStride*16;
  destination+=2048;
  valid=row+16<rows ? 16 : 0;
  asm volatile("ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  source+=kPipeTokenStride*16;
  destination+=2048;
  valid=row+32<rows ? 16 : 0;
  asm volatile("ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  source+=kPipeTokenStride*16;
  destination+=2048;
  valid=row+48<rows ? 16 : 0;
  asm volatile("ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#else
  (void)destination_base;(void)source_base;(void)first;(void)rows;
  (void)head;(void)feature_base;
#endif
}

CUTE_DEVICE void issue_pipe_v8_compact(
    uint16_t* destination_base,uint16_t const* source_base,
    int64_t first,int rows,int head,int value_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int tid=threadIdx.x;
  int row=(tid&120)>>3;
  int value=4*(tid&7);
  auto const* source=source_base+
      token_offset(first+row,head,value_base+value);
  int byte_offset=(tid<<3)^((tid&48)>>1);
  uint32_t destination=static_cast<uint32_t>(__cvta_generic_to_shared(
      reinterpret_cast<unsigned char*>(destination_base)+byte_offset));
  int valid=row<rows ? 8 : 0;
  asm volatile("ppu.cp.async.ca.shared.global [%0 + 0], [%1 + 0], 0x8, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  source+=kPipeTokenStride*16;
  destination+=1024;
  valid=row+16<rows ? 8 : 0;
  asm volatile("ppu.cp.async.ca.shared.global [%0 + 0], [%1 + 0], 0x8, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  source+=kPipeTokenStride*16;
  destination+=1024;
  valid=row+32<rows ? 8 : 0;
  asm volatile("ppu.cp.async.ca.shared.global [%0 + 0], [%1 + 0], 0x8, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  source+=kPipeTokenStride*16;
  destination+=1024;
  valid=row+48<rows ? 8 : 0;
  asm volatile("ppu.cp.async.ca.shared.global [%0 + 0], [%1 + 0], 0x8, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid) : "memory");
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#else
  (void)destination_base;(void)source_base;(void)first;(void)rows;
  (void)head;(void)value_base;
#endif
}

CUTE_DEVICE uint16_t* pipe_base(Stage& stage) {
  return reinterpret_cast<uint16_t*>(&stage);
}

template<int SegmentOffset>
CUTE_DEVICE void issue_pipe_tile16_segment(
    uint16_t* destination_base,uint16_t const* source_base,
    int64_t first,int rows,int head,int feature_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int segment=threadIdx.x+SegmentOffset;
  int row=segment/8;
  int feature=8*(segment%8);
  auto const* source=source_base+
      token_offset(first+row,head,feature_base+feature);
  uint32_t destination=static_cast<uint32_t>(__cvta_generic_to_shared(
      destination_base+swizzled(row,feature)));
  int valid_bytes=row<rows ? 16 : 0;
  asm volatile(
      "ppu.cp.async.cg.shared.global [%0 + 0], [%1 + 0], 0x10, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid_bytes) : "memory");
#else
  (void)destination_base;(void)source_base;(void)first;(void)rows;
  (void)head;(void)feature_base;
#endif
}

CUTE_DEVICE void issue_pipe_tile16(
    uint16_t* destination,uint16_t const* source,int64_t first,int rows,
    int head,int feature_base) {
  issue_pipe_tile16_segment<0>(
      destination,source,first,rows,head,feature_base);
  issue_pipe_tile16_segment<128>(
      destination,source,first,rows,head,feature_base);
  issue_pipe_tile16_segment<256>(
      destination,source,first,rows,head,feature_base);
  issue_pipe_tile16_segment<384>(
      destination,source,first,rows,head,feature_base);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

template<int RowOffset>
CUTE_DEVICE void issue_pipe_v8_segment(
    uint16_t* destination_base,uint16_t const* source_base,
    int64_t first,int rows,int head,int value_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  int tid=threadIdx.x;
  int row=((tid&120)>>3)+RowOffset;
  int value=4*(tid&7);
  auto const* source=source_base+
      token_offset(first+row,head,value_base+value);
  int byte_offset=(tid<<3)^((tid&48)>>1);
  byte_offset+=64*RowOffset;
  uint32_t destination=static_cast<uint32_t>(__cvta_generic_to_shared(
      reinterpret_cast<unsigned char*>(destination_base)+byte_offset));
  int valid_bytes=row<rows ? 8 : 0;
  asm volatile(
      "ppu.cp.async.ca.shared.global [%0 + 0], [%1 + 0], 0x8, %2;\n"
      :: "r"(destination),"l"(source),"r"(valid_bytes) : "memory");
#else
  (void)destination_base;(void)source_base;(void)first;(void)rows;
  (void)head;(void)value_base;
#endif
}

CUTE_DEVICE void issue_pipe_v8(
    uint16_t* destination,uint16_t const* source,int64_t first,int rows,
    int head,int value_base) {
  issue_pipe_v8_segment<0>(destination,source,first,rows,head,value_base);
  issue_pipe_v8_segment<16>(destination,source,first,rows,head,value_base);
  issue_pipe_v8_segment<32>(destination,source,first,rows,head,value_base);
  issue_pipe_v8_segment<48>(destination,source,first,rows,head,value_base);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

template<bool Trans,bool Swizzle>
CUTE_DEVICE void issue_pipe_aiu_tile64(
    uint16_t* destination,uint16_t const* source,int64_t first,int rows,
    int head,int feature_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) {
    using AiuLoad=PPU0010_AIU_LOAD<
        C<64*64*16>,bfloat16_t,Trans,Swizzle>;
    auto const* input=source+token_offset(first,head,feature_base);
    AiuDesc desc;
    if constexpr (Trans) {
      desc.template init<bfloat16_t,true,64,64>(
          reinterpret_cast<uint8_t const*>(input),64,rows,
          make_stride(_1{},FLA_CUTE_INT(kHeads*kDim)));
    } else {
      desc.template init<bfloat16_t,false,64,64>(
          reinterpret_cast<uint8_t const*>(input),rows,64,
          make_stride(FLA_CUTE_INT(kHeads*kDim),_1{}));
    }
    uint32_t output=static_cast<uint32_t>(
        __cvta_generic_to_shared(destination));
    AiuLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
                  reinterpret_cast<void const*>(input),desc,0,0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  }
#else
  (void)destination;(void)source;(void)first;(void)rows;
  (void)head;(void)feature_base;
#endif
}

CUTE_DEVICE void issue_pipe_aiu_v(
    uint16_t* destination,uint16_t const* source,int64_t first,int rows,
    int head,int value_base) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  if(threadIdx.x==0) {
    using AiuLoad=PPU0010_AIU_LOAD<
        C<64*kBv*16>,bfloat16_t,false,false>;
    auto const* input=source+token_offset(first,head,value_base);
    AiuDesc desc;
    desc.template init<bfloat16_t,false,64,kBv>(
        reinterpret_cast<uint8_t const*>(input),rows,kBv,
        make_stride(FLA_CUTE_INT(kHeads*kDim),_1{}));
    uint32_t output=static_cast<uint32_t>(
        __cvta_generic_to_shared(destination));
    AiuLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
                  reinterpret_cast<void const*>(input),desc,0,0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  }
#else
  (void)destination;(void)source;(void)first;(void)rows;
  (void)head;(void)value_base;
#endif
}

CUTE_DEVICE void issue_pipe_chunk(
    Stage& stage,uint16_t const* k,uint16_t const* v,uint16_t const* w,
    int64_t first,int rows,int head,int value_base,bool include_k1) {
  auto* base=pipe_base(stage);
#if FLA_RECURRENCE_PIPELINE5_AIU
  issue_pipe_aiu_tile64<false,true>(base+kPipeW0,w,first,rows,head,0);
  issue_pipe_aiu_tile64<false,true>(base+kPipeW1,w,first,rows,head,64);
#if FLA_RECURRENCE_DIRECT_V_LOAD
  if(rows<kBt) issue_pipe_aiu_v(base+kPipeV,v,first,rows,head,value_base);
#else
  issue_pipe_aiu_v(base+kPipeV,v,first,rows,head,value_base);
#endif
  issue_pipe_aiu_tile64<true,true>(base+kPipeK0,k,first,rows,head,0);
  if(include_k1)
    issue_pipe_aiu_tile64<true,true>(base+kPipeK1,k,first,rows,head,64);
#else
#if FLA_RECURRENCE_PIPELINE5_COMPACT_CPASYNC
  issue_pipe_tile16_compact(base+kPipeW0,w,first,rows,head,0);
  issue_pipe_tile16_compact(base+kPipeW1,w,first,rows,head,64);
  issue_pipe_v8_compact(base+kPipeV,v,first,rows,head,value_base);
  issue_pipe_tile16_compact(base+kPipeK0,k,first,rows,head,0);
  if(include_k1)
    issue_pipe_tile16_compact(base+kPipeK1,k,first,rows,head,64);
#else
  issue_pipe_tile16(base+kPipeW0,w,first,rows,head,0);
  issue_pipe_tile16(base+kPipeW1,w,first,rows,head,64);
  issue_pipe_v8(base+kPipeV,v,first,rows,head,value_base);
  issue_pipe_tile16(base+kPipeK0,k,first,rows,head,0);
  if(include_k1) issue_pipe_tile16(base+kPipeK1,k,first,rows,head,64);
#endif
#endif
}

CUTE_DEVICE void issue_pipe_k1(
    Stage& stage,uint16_t const* k,int64_t first,int rows,int head) {
#if FLA_RECURRENCE_PIPELINE5_AIU
  issue_pipe_aiu_tile64<true,true>(
      pipe_base(stage)+kPipeK1,k,first,rows,head,64);
#else
#if FLA_RECURRENCE_PIPELINE5_COMPACT_CPASYNC
  issue_pipe_tile16_compact(
      pipe_base(stage)+kPipeK1,k,first,rows,head,64);
#else
  issue_pipe_tile16(pipe_base(stage)+kPipeK1,k,first,rows,head,64);
#endif
#endif
}

CUTE_DEVICE void wait_pipe_chunk() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
#if FLA_RECURRENCE_PIPELINE5_AIU_LEADER_WAIT
  if(threadIdx.x==0) asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#else
  asm volatile("ppu.cp.async.wait_group 0;\n" ::: "memory");
#endif
#endif
}

template<class ACopyAtom,class Accumulator>
CUTE_DEVICE void mma_accumulate_pipe_w(
    uint16_t* a_storage,uint16_t* b_storage,Mma const& mma,
    Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(a_storage))));
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(b_storage))));
  auto a=make_tensor(ap,ALogical{});
  auto b=make_tensor(bp,BLogical{});
  auto thr=Mma{}.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  auto r_b=thr.partition_fragment_B(b);
  auto copy_a=make_tiled_copy_A(ACopyAtom{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  auto copy_b=make_tiled_copy_B(BReader{},mma);
  auto d_b=copy_b.get_thread_slice(tid).retile_D(r_b);
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    CUTE_UNROLL
    for(int fragment=0;fragment<kBv/16;++fragment) {
      auto src=b_source_slice(bp,tid,fragment,q);
      auto dst=d_b(_,fragment,q);
      BReader{}.call(src,dst);
    }
    cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
  }
}

template<class ACopyAtom,class BFragment,class Accumulator>
CUTE_DEVICE void mma_accumulate_pipe_loaded_b(
    uint16_t* a_storage,Mma const& mma,BFragment const& b_fragment,
    Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(a_storage))));
  auto a=make_tensor(ap,ALogical{});
  auto thr=mma.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  auto copy_a=make_tiled_copy_A(ACopyAtom{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    cute::gemm(mma,r_a(_,_,q),b_fragment(_,_,q),accumulator);
  }
}

template<class BFragment>
CUTE_DEVICE void load_pipe_b_fragment(
    uint16_t* b_storage,Mma const& mma,BFragment& fragment) {
  int tid=threadIdx.x;
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(b_storage))));
  auto copy_b=make_tiled_copy_B(BReader{},mma);
  auto d_b=copy_b.get_thread_slice(tid).retile_D(fragment);
  CUTE_UNROLL
  for(int q=0;q<4;++q)
    CUTE_UNROLL
    for(int part=0;part<kBv/16;++part) {
      auto src=b_source_slice(bp,tid,part,q);
      auto dst=d_b(_,part,q);
      BReader{}.call(src,dst);
    }
}

template<class BFragment>
CUTE_DEVICE void load_pipe_b_fragment_reference(
    uint16_t* b_storage,Mma const&,BFragment& fragment) {
  int tid=threadIdx.x;
  int address0=((tid&7)<<5)|((tid&6)<<2);
  address0^=tid&8;
  address0|=(tid&16)<<4;
  uint32_t shared_base=static_cast<uint32_t>(
      __cvta_generic_to_shared(b_storage));
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    CUTE_UNROLL
    for(int part=0;part<kBv/16;++part) {
      auto destination=fragment(_,part,q);
      auto* words=reinterpret_cast<uint32_t*>(
          cute::raw_pointer_cast(destination.data()));
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
      uint32_t address=shared_base+(part==0
          ? 2*address0+1024*q : 2*(address0^(16+512*q)));
      asm volatile(
          "ppu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 "
          "{%0, %1, %2, %3}, [%4];"
          : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3])
          : "r"(address));
#endif
    }
  }
}

template<class ACopyAtom,class Accumulator>
CUTE_DEVICE void mma_accumulate_pipe_w_reference(
    uint16_t* a_storage,uint16_t* b_storage,Mma const& mma,
    Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(a_storage))));
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(b_storage))));
  auto thr=mma.get_thread_slice(tid);
  auto a=make_tensor(ap,ALogical{});
  auto b=make_tensor(bp,BLogical{});
  auto r_a=thr.partition_fragment_A(a);
  auto r_b=thr.partition_fragment_B(b);
  auto copy_a=make_tiled_copy_A(ACopyAtom{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  load_pipe_b_fragment_reference(b_storage,mma,r_b);
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
  }
}

template<class BFragment,class Accumulator>
CUTE_DEVICE void mma_accumulate_pipe_k(
    uint16_t* k_storage,Mma const& mma,BFragment const& b_fragment,
    Accumulator& accumulator) {
  int tid=threadIdx.x;
  auto dummy=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(k_storage))));
  auto a=make_tensor(dummy,ALogical{});
  auto r_a=mma.get_thread_slice(tid).partition_fragment_A(a);
  int address0=(((tid&7)<<6)|((tid&7)<<3))^
               (((tid&96)>>1)|(tid&8));
  address0|=(tid&16)<<5;
  uint32_t shared_base=static_cast<uint32_t>(
      __cvta_generic_to_shared(k_storage));
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    auto fragment=r_a(_,_,q);
    auto* words=reinterpret_cast<uint32_t*>(
        cute::raw_pointer_cast(fragment.data()));
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    uint32_t address=shared_base+2*address0+2048*q;
    asm volatile(
        "ppu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3])
        : "r"(address));
#endif
    cute::gemm(mma,r_a(_,_,q),b_fragment(_,_,q),accumulator);
  }
}

CUTE_DEVICE void store_pipe_history_and_state(
    uint16_t* b_storage,uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  int tid=threadIdx.x;
  int lane=tid%32;
  int warp=tid/32;
  uint16_t state_bf16[kStateOrdinals];
  CUTE_UNROLL
  for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
    int value=recurrence_state_v(tid,ordinal);
    int key=recurrence_state_k(tid,ordinal);
    uint16_t rounded=round_bf16(carried[panel][ordinal]);
    state_bf16[ordinal]=rounded;
    b_storage[swizzled(key,value)]=rounded;
  }
  CUTE_UNROLL
  for(int value_group=0;value_group<kBv/4;++value_group) {
    int ordinal_low=(value_group&3)+((value_group&4)<<1);
    int ordinal_high=ordinal_low+4;
    unsigned source=unsigned(state_bf16[ordinal_low]) |
                    (unsigned(state_bf16[ordinal_high])<<16);
    Bf16Pack8 low{};
    Bf16Pack8 high{};
    CUTE_UNROLL
    for(int key=0;key<8;++key) {
      int source_lane=(lane&3)+4*key;
      unsigned packed=__shfl_sync(0xffffffffu,source,source_lane,32);
      low.element[key]=uint16_t(packed);
      high.element[key]=uint16_t(packed>>16);
    }
    if(lane<4) {
      int value=4*value_group+lane;
      int64_t output=history_offset(
          history_chunk,head,value_base+value,64*panel+16*warp);
      *reinterpret_cast<uint4*>(h+output)=low.vector;
      *reinterpret_cast<uint4*>(h+output+8)=high.vector;
    }
  }
}

CUTE_DEVICE void store_pipe_history_and_state_scratch(
    uint16_t* b_storage,uint16_t* history_scratch,uint16_t* h,
    float const (&carried)[2][kStateOrdinals],int panel,int64_t history_chunk,int head,
    int value_base) {
  int tid=threadIdx.x;
  CUTE_UNROLL
  for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
    int value=recurrence_state_v(tid,ordinal);
    int key=recurrence_state_k(tid,ordinal);
    uint16_t rounded=round_bf16(carried[panel][ordinal]);
    b_storage[swizzled(key,value)]=rounded;
    history_scratch[value*64+key]=rounded;
  }
  __syncthreads();
  int linear=16*tid;
  int value=linear/64;
  int key=linear%64;
  auto output=h+history_offset(
      history_chunk,head,value_base+value,64*panel+key);
  *reinterpret_cast<uint4*>(output)=
      *reinterpret_cast<uint4 const*>(history_scratch+linear);
  *reinterpret_cast<uint4*>(output+8)=
      *reinterpret_cast<uint4 const*>(history_scratch+linear+8);
}

CUTE_DEVICE uint32_t round_bf16_pair(float low,float high) {
  return uint32_t(round_bf16(low))|(uint32_t(round_bf16(high))<<16);
}

CUTE_DEVICE uint32_t round_bf16_pair_packed(float low,float high) {
  union {
    uint32_t word;
    __ppu_bfloat162 pair;
  } result;
  result.pair=__floats2bfloat162_rn(low,high);
  return result.word;
}

// The reference history image is row-major [value,key], while the live state
// fragment is distributed as four 32-lane key groups.  Transpose that fragment
// directly through the warp crossbar.  This preserves the exact history bytes
// but avoids the reference path's first scratch publish, eight scratch reloads
// per thread, and the intervening CTA barrier.  The reference MMA image is
// still rebuilt below, so this experiment changes no MMA operand layout.
CUTE_DEVICE void store_pipe_history_reference_shfl(
    uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  static_assert(kBv==16 || FLA_RECURRENCE_BODY_ONLY);
  int tid=threadIdx.x;
  int lane=tid&31;
  int warp=tid>>5;
  CUTE_UNROLL
  for(int value_group=0;value_group<4;++value_group) {
    int ordinal_low=value_group;
    int ordinal_high=value_group+4;
    unsigned source=unsigned(round_bf16(carried[panel][ordinal_low])) |
                    (unsigned(round_bf16(carried[panel][ordinal_high]))<<16);
    Bf16Pack8 low{};
    Bf16Pack8 high{};
    CUTE_UNROLL
    for(int key=0;key<8;++key) {
      int source_lane=(lane&3)+4*key;
      unsigned packed=__shfl_sync(0xffffffffu,source,source_lane,32);
      low.element[key]=uint16_t(packed);
      high.element[key]=uint16_t(packed>>16);
    }
    if(lane<4) {
      int value=4*value_group+lane;
      int64_t output=history_offset(
          history_chunk,head,value_base+value,64*panel+16*warp);
      *reinterpret_cast<uint4*>(h+output)=low.vector;
      *reinterpret_cast<uint4*>(h+output+8)=high.vector;
    }
  }
}

CUTE_DEVICE void store_pipe_history_and_state_reference(
    uint16_t* scratch,uint16_t* h,float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  int tid=threadIdx.x;
  auto* bytes=reinterpret_cast<uint8_t*>(scratch);
#if FLA_RECURRENCE_DIRECT_HISTORY_SHFL
  store_pipe_history_reference_shfl(
      h,carried,panel,history_chunk,head,value_base);
#else
  int store_base=(((tid&24)<<2)^(tid&100))|
                 ((tid&28)<<7)|((tid&3)<<3);
#if FLA_RECURRENCE_BV16
  *reinterpret_cast<uint32_t*>(bytes+store_base)=
      uint32_t(round_bf16(carried[panel][0]));
  *reinterpret_cast<uint32_t*>(bytes+store_base+128)=
      uint32_t(round_bf16(carried[panel][1]));
  *reinterpret_cast<uint32_t*>(bytes+store_base+256)=
      uint32_t(round_bf16(carried[panel][2]));
  *reinterpret_cast<uint32_t*>(bytes+store_base+384)=
      uint32_t(round_bf16(carried[panel][3]));
  int high_base=store_base^4;
  *reinterpret_cast<uint32_t*>(bytes+high_base)=
      uint32_t(round_bf16(carried[panel][4]));
  *reinterpret_cast<uint32_t*>(bytes+high_base+128)=
      uint32_t(round_bf16(carried[panel][5]));
  *reinterpret_cast<uint32_t*>(bytes+high_base+256)=
      uint32_t(round_bf16(carried[panel][6]));
  *reinterpret_cast<uint32_t*>(bytes+high_base+384)=
      uint32_t(round_bf16(carried[panel][7]));
#else
  *reinterpret_cast<uint32_t*>(bytes+store_base)=
      round_bf16_pair(carried[panel][0],carried[panel][8]);
  *reinterpret_cast<uint32_t*>(bytes+store_base+128)=
      round_bf16_pair(carried[panel][1],carried[panel][9]);
  *reinterpret_cast<uint32_t*>(bytes+store_base+256)=
      round_bf16_pair(carried[panel][2],carried[panel][10]);
  *reinterpret_cast<uint32_t*>(bytes+store_base+384)=
      round_bf16_pair(carried[panel][3],carried[panel][11]);
  int high_base=store_base^4;
  *reinterpret_cast<uint32_t*>(bytes+high_base)=
      round_bf16_pair(carried[panel][4],carried[panel][12]);
  *reinterpret_cast<uint32_t*>(bytes+high_base+128)=
      round_bf16_pair(carried[panel][5],carried[panel][13]);
  *reinterpret_cast<uint32_t*>(bytes+high_base+256)=
      round_bf16_pair(carried[panel][6],carried[panel][14]);
  *reinterpret_cast<uint32_t*>(bytes+high_base+384)=
      round_bf16_pair(carried[panel][7],carried[panel][15]);
#endif
  __syncthreads();

  int load_base=((tid<<2)&388)|(tid&24)|((tid&6)<<4);
  uint32_t p0=*reinterpret_cast<uint32_t const*>(bytes+load_base);
  uint32_t p1=*reinterpret_cast<uint32_t const*>(bytes+(load_base^516));
  uint32_t p2=*reinterpret_cast<uint32_t const*>(bytes+(load_base^1056));
  uint32_t p3=*reinterpret_cast<uint32_t const*>(bytes+(load_base^1572));
  uint32_t p4=*reinterpret_cast<uint32_t const*>(bytes+(load_base^2112));
  uint32_t p5=*reinterpret_cast<uint32_t const*>(bytes+(load_base^2628));
  uint32_t p6=*reinterpret_cast<uint32_t const*>(bytes+(load_base^3168));
  uint32_t p7=*reinterpret_cast<uint32_t const*>(bytes+(load_base^3684));
  Bf16Pack8 low{};
#if !FLA_RECURRENCE_BV16
  Bf16Pack8 high{};
#endif
#if FLA_RECURRENCE_BV16
#define FLA_UNPACK_HISTORY_PAIR(I, P) do { \
    low.element[I]=uint16_t(P); \
  } while(false)
#else
#define FLA_UNPACK_HISTORY_PAIR(I, P) do { \
    low.element[I]=uint16_t(P); \
    high.element[I]=uint16_t((P)>>16); \
  } while(false)
#endif
  FLA_UNPACK_HISTORY_PAIR(0,p0);
  FLA_UNPACK_HISTORY_PAIR(1,p1);
  FLA_UNPACK_HISTORY_PAIR(2,p2);
  FLA_UNPACK_HISTORY_PAIR(3,p3);
  FLA_UNPACK_HISTORY_PAIR(4,p4);
  FLA_UNPACK_HISTORY_PAIR(5,p5);
  FLA_UNPACK_HISTORY_PAIR(6,p6);
  FLA_UNPACK_HISTORY_PAIR(7,p7);
#undef FLA_UNPACK_HISTORY_PAIR
  int value=(tid&120)>>3;
  int key=(tid&7)<<3;
  *reinterpret_cast<uint4*>(h+history_offset(
      history_chunk,head,value_base+value,64*panel+key))=low.vector;
#if !FLA_RECURRENCE_BV16
  *reinterpret_cast<uint4*>(h+history_offset(
      history_chunk,head,value_base+value+16,64*panel+key))=high.vector;
#endif
  __syncthreads();
#endif

  int bbase=((tid&96)<<5)|((tid&28)<<4)|((tid<<1)&54);
  // Recreate Triton's state-MMA image from the still-live register fragment.
  // Reloading these values from `scratch` while overwriting the same 4 KiB
  // image creates a cross-thread read/write race.
  *reinterpret_cast<uint16_t*>(bytes+bbase)=
      round_bf16(carried[panel][0]);
  *reinterpret_cast<uint16_t*>(bytes+bbase+8)=
      round_bf16(carried[panel][1]);
  *reinterpret_cast<uint16_t*>(bytes+bbase+512)=
      round_bf16(carried[panel][4]);
  *reinterpret_cast<uint16_t*>(bytes+bbase+520)=
      round_bf16(carried[panel][5]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16))=
      round_bf16(carried[panel][2]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+8)=
      round_bf16(carried[panel][3]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+512)=
      round_bf16(carried[panel][6]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+520)=
      round_bf16(carried[panel][7]);
#if !FLA_RECURRENCE_BV16
  *reinterpret_cast<uint16_t*>(bytes+(bbase^32))=
      round_bf16(carried[panel][8]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^32)+8)=
      round_bf16(carried[panel][9]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^32)+512)=
      round_bf16(carried[panel][12]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^32)+520)=
      round_bf16(carried[panel][13]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^48))=
      round_bf16(carried[panel][10]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^48)+8)=
      round_bf16(carried[panel][11]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^48)+512)=
      round_bf16(carried[panel][14]);
  *reinterpret_cast<uint16_t*>(bytes+(bbase^48)+520)=
      round_bf16(carried[panel][15]);
#endif
}

CUTE_DEVICE void store_pipe_history_and_state_output_native(
    uint16_t* b_storage,uint16_t* h,
    float const (&carried)[2][kStateOrdinals],int panel,
    int64_t history_chunk,int head,int value_base) {
  static_assert(kBv==16 || FLA_RECURRENCE_BODY_ONLY);
  int tid=threadIdx.x;
  Bf16Pack8 packed{};
  CUTE_UNROLL
  for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
    uint16_t rounded=round_bf16(carried[panel][ordinal]);
    packed.element[ordinal]=rounded;
  }
  // Preserve the already-proved 4-warp recurrence B image for the W*state
  // MMA.  Only the history output changes to the packed register-owner image;
  // mixing in the 8-warp output reader here changes the fragment mapping.
  auto* bytes=reinterpret_cast<uint8_t*>(b_storage);
  int bbase=((tid&96)<<5)|((tid&28)<<4)|((tid<<1)&54);
  *reinterpret_cast<uint16_t*>(bytes+bbase)=packed.element[0];
  *reinterpret_cast<uint16_t*>(bytes+bbase+8)=packed.element[1];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16))=packed.element[2];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+8)=packed.element[3];
  *reinterpret_cast<uint16_t*>(bytes+bbase+512)=packed.element[4];
  *reinterpret_cast<uint16_t*>(bytes+bbase+520)=packed.element[5];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+512)=packed.element[6];
  *reinterpret_cast<uint16_t*>(bytes+(bbase^16)+520)=packed.element[7];
#if FLA_RECURRENCE_BFRAGMENT_HISTORY
  // Only the global history ABI changes. The recurrence B scratch above
  // and all FP32 carried values remain in the frozen build48 ownership.
  int lane=tid%32;
  uint32_t selector=(lane&4) ? 0x3276u : 0x5410u;
  uint32_t x=__shfl_xor_sync(0xffffffffu,packed.vector.x,4,32);
  uint32_t y=__shfl_xor_sync(0xffffffffu,packed.vector.y,4,32);
  uint32_t z=__shfl_xor_sync(0xffffffffu,packed.vector.z,4,32);
  uint32_t w=__shfl_xor_sync(0xffffffffu,packed.vector.w,4,32);
  uint4 published;
  published.x=__byte_perm(packed.vector.x,x,selector);
  published.y=__byte_perm(packed.vector.z,z,selector);
  published.z=__byte_perm(packed.vector.y,y,selector);
  published.w=__byte_perm(packed.vector.w,w,selector);
  int record_lane=(lane>>3)+4*(lane&7);
  auto* output=h+output_native_history_offset(
      history_chunk,head,value_base,panel,8*(32*(tid/32)+record_lane));
  *reinterpret_cast<uint4*>(output)=published;
#else
  auto* output=h+output_native_history_offset(
      history_chunk,head,value_base,panel,8*tid);
  *reinterpret_cast<uint4*>(output)=packed.vector;
#endif
}

CUTE_DEVICE int pipe_v_element_offset(int row,int value) {
  int tid=8*(row&15)+(value>>2);
  int byte_offset=(tid<<3)^((tid&48)>>1);
  byte_offset+=(row>>4)*1024;
  return (byte_offset>>1)+(value&3);
}

#if FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL
template<class Accumulator>
CUTE_DEVICE void publish_pipe_residual_direct_shfl(
    Stage& stage,uint16_t const* v,uint16_t* v_new,Accumulator const& residual,
    int64_t first,int head,int value_base) {
  static_assert(kBv==16 || FLA_RECURRENCE_BODY_ONLY);
  float fragment[8];
  int slot=0;
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(residual));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(residual));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(residual));++value_index)
        fragment[slot++]=residual(value_index,m,n);

  // Exact tc01 M64N16 C layout, established by the physical coordinate probe:
  // each four-lane group owns one row; slots 0..3 cover values lane+{0,4,8,12}
  // and slots 4..7 cover the row eight positions later.  Lanes 0 and 1 gather
  // one contiguous BF16x8 segment each, avoiding the residual TSM round-trip.
  int lane=threadIdx.x&31;
  int source_base=lane&~3;
  float gathered[2][8];
  CUTE_UNROLL
  for(int output_half=0;output_half<2;++output_half) {
    CUTE_UNROLL
    for(int row_half=0;row_half<2;++row_half) {
      int fragment_base=4*row_half+2*output_half;
      CUTE_UNROLL
      for(int source=0;source<4;++source) {
        gathered[row_half][source]=__shfl_sync(
            0xffffffffu,fragment[fragment_base],source_base+source,32);
        gathered[row_half][source+4]=__shfl_sync(
            0xffffffffu,fragment[fragment_base+1],source_base+source,32);
      }
    }
    if((lane&3)==output_half) {
      auto* base=pipe_base(stage);
      auto* b_storage=base+kPipeW1;
      auto* v_storage=base+kPipeV;
      int row_base=16*(threadIdx.x>>5)+(lane>>2);
      int value=8*output_half;
      CUTE_UNROLL
      for(int row_half=0;row_half<2;++row_half) {
        int row=row_base+8*row_half;
#if FLA_RECURRENCE_DIRECT_V_LOAD
        Bf16Pack8 input{};
        input.vector=*reinterpret_cast<uint4 const*>(
            v+token_offset(first+row,head,value_base+value));
#endif
        Bf16Pack8 output{};
#if FLA_RECURRENCE_PACKED_RESIDUAL_CONVERT
        CUTE_UNROLL
        for(int i=0;i<4;++i) {
          float low=__fsub_rn(
#if FLA_RECURRENCE_DIRECT_V_LOAD
              load_bf16(input.element[2*i]),gathered[row_half][2*i]);
#else
              load_bf16(v_storage[row*kBv+value+2*i]),
              gathered[row_half][2*i]);
#endif
          float high=__fsub_rn(
#if FLA_RECURRENCE_DIRECT_V_LOAD
              load_bf16(input.element[2*i+1]),gathered[row_half][2*i+1]);
#else
              load_bf16(v_storage[row*kBv+value+2*i+1]),
              gathered[row_half][2*i+1]);
#endif
          output.pair[i]=round_bf16_pair_packed(low,high);
        }
#else
        CUTE_UNROLL
        for(int i=0;i<8;++i)
          output.element[i]=round_bf16(__fsub_rn(
#if FLA_RECURRENCE_DIRECT_V_LOAD
              load_bf16(input.element[i]),gathered[row_half][i]));
#else
              load_bf16(v_storage[row*kBv+value+i]),gathered[row_half][i]));
#endif
#endif
        *reinterpret_cast<uint4*>(b_storage+swizzled(row,value))=output.vector;
        *reinterpret_cast<uint4*>(
            v_new+token_offset(first+row,head,value_base+value))=output.vector;
      }
    }
  }
}
#endif

template<bool FullRows,class Accumulator,class Coordinates>
CUTE_DEVICE void publish_pipe_residual(
    Stage& stage,uint16_t const* v,uint16_t* v_new,Accumulator const& residual,
    Coordinates const& coordinates,int64_t first,int rows,int head,
    int value_base) {
#if FLA_RECURRENCE_DIRECT_RESIDUAL_SHFL
  if constexpr (FullRows) {
    publish_pipe_residual_direct_shfl(
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
      uint16_t input=
#if FLA_RECURRENCE_PIPELINE5_AIU
          v_storage[row*kBv+value+i];
#else
          v_storage[pipe_v_element_offset(row,value+i)];
#endif
      if constexpr (FullRows) {
        output.element[i]=round_bf16(__fsub_rn(
            load_bf16(input),residual_storage[row*kBv+value+i]));
      } else {
        output.element[i]=row<rows ? round_bf16(__fsub_rn(
            load_bf16(input),residual_storage[row*kBv+value+i])) : uint16_t(0);
      }
    }
    if constexpr (FullRows) {
      *reinterpret_cast<uint4*>(
          v_new+token_offset(first+row,head,value_base+value))=output.vector;
    } else {
      if(row<rows) *reinterpret_cast<uint4*>(
          v_new+token_offset(first+row,head,value_base+value))=output.vector;
    }
    *reinterpret_cast<uint4*>(b_storage+swizzled(row,value))=output.vector;
  }
}

template<int NtBucket>
CUTE_DEVICE void recurrence_body_pipeline5(
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,float const* decay_cache,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T,Stage& stage) {
  int tid=threadIdx.x;
  int value_base=blockIdx.x*kBv;
  int head=blockIdx.y;
  int sequence=blockIdx.z;
  int bos=cu_seqlens[sequence],eos=cu_seqlens[sequence+1];
  if(bos<0 || eos>T || eos<bos) return;
  int nt=(eos-bos+kBt-1)/kBt;
  int64_t history_base=chunk_offsets[sequence];
  int slot=state_indices ? state_indices[sequence] : sequence;
  float carried[2][kStateOrdinals];
  CUTE_UNROLL
  for(int panel=0;panel<2;++panel)
    CUTE_UNROLL
    for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
      int value=recurrence_state_v(tid,ordinal);
      int key=recurrence_state_k(tid,ordinal);
      int64_t offset=state_offset(slot,head,value_base+value,64*panel+key);
      float raw=0.0f;
      if(initial_state) {
        raw=state_is_fp32
            ? load_bf16(round_bf16(static_cast<float const*>(initial_state)[offset]))
            : load_bf16(static_cast<uint16_t const*>(initial_state)[offset]);
      }
      carried[panel][ordinal]=__fadd_rn(raw,0.0f);
    }

  Mma mma;
  auto thr=mma.get_thread_slice(tid);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},Int<kBv>{})));
  auto accumulator=Mma::make_fragment_C(coordinates);
  auto* base=pipe_base(stage);
  if(nt>0) {
    int rows0=min(kBt,eos-bos);
    issue_pipe_chunk(stage,k,v,w,bos,rows0,head,value_base,true);
  }

  for(int chunk=0;chunk<nt;++chunk) {
    int64_t first=bos+int64_t(chunk)*kBt;
    int rows=min(kBt,eos-int(first));
    auto* b_storage=base+kPipeScratch;
    clear(accumulator);
#if FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
    store_pipe_history_and_state_output_native(
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
    wait_pipe_chunk();
    __syncthreads();
#if FLA_RECURRENCE_PIPELINE5_AIU
#if FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
    mma_accumulate_pipe_w_reference<AiuAReader>(
        base+kPipeW0,b_storage,mma,accumulator);
#elif FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
    mma_accumulate_pipe_w_reference<AiuAReader>(
        base+kPipeW0,b_storage,mma,accumulator);
#else
    mma_accumulate_pipe_w<AiuAReader>(
        base+kPipeW0,b_storage,mma,accumulator);
#endif
#else
#if FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
    mma_accumulate_pipe_w_reference<AReader>(
        base+kPipeW0,b_storage,mma,accumulator);
#else
    mma_accumulate_pipe_w<AReader>(
        base+kPipeW0,b_storage,mma,accumulator);
#endif
#endif
    __syncthreads();
#if FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
    store_pipe_history_and_state_output_native(
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
    __syncthreads();
#if FLA_RECURRENCE_PIPELINE5_AIU
#if FLA_RECURRENCE_OUTPUT_NATIVE_HISTORY
    mma_accumulate_pipe_w_reference<AiuAReader>(
        base+kPipeW1,b_storage,mma,accumulator);
#elif FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
    mma_accumulate_pipe_w_reference<AiuAReader>(
        base+kPipeW1,b_storage,mma,accumulator);
#else
    mma_accumulate_pipe_w<AiuAReader>(
        base+kPipeW1,b_storage,mma,accumulator);
#endif
#else
#if FLA_RECURRENCE_REFERENCE_STATE_LAYOUT
    mma_accumulate_pipe_w_reference<AReader>(
        base+kPipeW1,b_storage,mma,accumulator);
#else
    mma_accumulate_pipe_w<AReader>(
        base+kPipeW1,b_storage,mma,accumulator);
#endif
#endif
    __syncthreads();
#if FLA_RECURRENCE_FULL_ROWS_SPECIALIZATION
    if(rows==kBt) {
      publish_pipe_residual<true>(
          stage,v,v_new,accumulator,coordinates,first,rows,head,value_base);
    } else {
      publish_pipe_residual<false>(
          stage,v,v_new,accumulator,coordinates,first,rows,head,value_base);
    }
#else
    publish_pipe_residual<false>(
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

    int last=min(first+kBt,int64_t(eos))-1;
    clear(accumulator);
#if FLA_RECURRENCE_PIPELINE5_AIU
    mma_accumulate_pipe_loaded_b<AiuATransReader>(
        base+kPipeK0,mma,b_fragment,accumulator);
#else
    mma_accumulate_pipe_k(base+kPipeK0,mma,b_fragment,accumulator);
#endif
    update_carried_from_fragment<
        FLA_RECURRENCE_PRECOMPUTE_DECAY && NtBucket==2>(
        carried,0,accumulator,gk,decay_cache,chunk,last,head);
    __syncthreads();
    bool has_next=chunk+1<nt;
    int64_t next_first=first+kBt;
    int next_rows=has_next ? min(kBt,eos-int(next_first)) : 0;
    if(has_next) issue_pipe_chunk(
        stage,k,v,w,next_first,next_rows,head,value_base,false);

    clear(accumulator);
#if FLA_RECURRENCE_PIPELINE5_AIU
    mma_accumulate_pipe_loaded_b<AiuATransReader>(
        base+kPipeK1,mma,b_fragment,accumulator);
#else
    mma_accumulate_pipe_k(base+kPipeK1,mma,b_fragment,accumulator);
#endif
    update_carried_from_fragment<
        FLA_RECURRENCE_PRECOMPUTE_DECAY && NtBucket==2>(
        carried,1,accumulator,gk,decay_cache,chunk,last,head);
    __syncthreads();
    if(has_next) issue_pipe_k1(stage,k,next_first,next_rows,head);
  }

  if(final_state) {
    CUTE_UNROLL
    for(int panel=0;panel<2;++panel)
      CUTE_UNROLL
      for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
        int value=recurrence_state_v(tid,ordinal);
        int key=recurrence_state_k(tid,ordinal);
        int64_t offset=state_offset(slot,head,value_base+value,64*panel+key);
        uint16_t rounded=round_bf16(carried[panel][ordinal]);
        if(state_is_fp32) static_cast<float*>(final_state)[offset]=load_bf16(rounded);
        else static_cast<uint16_t*>(final_state)[offset]=rounded;
      }
  }
}
#endif

template<int NtBucket>
CUTE_DEVICE void recurrence_body(
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T,Stage& stage) {
  static_assert(NtBucket>=0 && NtBucket<=2);
  int tid=threadIdx.x;
  int value_base=blockIdx.x*kBv;
  int head=blockIdx.y;
  int sequence=blockIdx.z;
  int bos=cu_seqlens[sequence],eos=cu_seqlens[sequence+1];
  if(bos<0 || eos>T || eos<bos) return;
  int nt=(eos-bos+kBt-1)/kBt;
  int64_t history_base=chunk_offsets[sequence];
  int slot=state_indices ? state_indices[sequence] : sequence;
  float carried[2][kStateOrdinals];
  CUTE_UNROLL
  for(int panel=0;panel<2;++panel) {
    CUTE_UNROLL
    for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
      int value=recurrence_state_v(tid,ordinal);
      int key=recurrence_state_k(tid,ordinal);
      int64_t offset=state_offset(slot,head,value_base+value,64*panel+key);
      float raw=0.0f;
      if(initial_state) {
        raw=state_is_fp32
            ? load_bf16(round_bf16(static_cast<float const*>(initial_state)[offset]))
            : load_bf16(static_cast<uint16_t const*>(initial_state)[offset]);
      }
      carried[panel][ordinal]=__fadd_rn(raw,0.0f);
    }
  }

  Mma mma;
  auto thr=mma.get_thread_slice(tid);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},Int<kBv>{})));
  auto residual=Mma::make_fragment_C(coordinates);
  CUTE_STATIC_ASSERT_V(size<0>(residual)==Int<8>{});
  CUTE_STATIC_ASSERT_V(size<1>(residual)==Int<1>{});
  CUTE_STATIC_ASSERT_V(size<2>(residual)==Int<kBv/16>{});

  for(int chunk=0;chunk<nt;++chunk) {
    int64_t first=bos+int64_t(chunk)*kBt;
    int rows=min(kBt,eos-int(first));
    clear(residual);
    CUTE_UNROLL
    for(int panel=0;panel<2;++panel) {
      if(panel!=0) __syncthreads();
#if FLA_RECURRENCE_OVERLAP_AIU
      issue_w_panel_aiu(stage,w,first,rows,head,panel);
      store_history_and_publish_state(
          stage,h,carried,panel,history_base+chunk,head,value_base);
      wait_panel_aiu();
#else
      publish_w_panel(stage,w,first,rows,head,panel);
      store_history_and_publish_state(
          stage,h,carried,panel,history_base+chunk,head,value_base);
#endif
      __syncthreads();
#if FLA_RECURRENCE_AIU_W
      mma_accumulate<AiuAReader>(stage,mma,residual);
#else
      mma_accumulate<AReader>(stage,mma,residual);
#endif
    }
    // All W/state readers finish before B is reused for the rounded residual.
    __syncthreads();
#if FLA_RECURRENCE_OVERLAP_AIU
#if FLA_RECURRENCE_CPASYNC_K
    issue_kg_panel_cpasync(stage,k,first,rows,head,0);
#else
    issue_kg_panel_aiu(stage,k,first,rows,head,0);
#endif
#endif
    publish_residual(
        stage,v,v_new,residual,coordinates,first,rows,head,value_base);
#if FLA_RECURRENCE_OVERLAP_AIU
#if FLA_RECURRENCE_CPASYNC_K
    wait_kg_panel_cpasync();
#else
    wait_panel_aiu();
#endif
#endif
    __syncthreads();
#if FLA_RECURRENCE_REUSE_K_B
    auto k_update_bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
        static_cast<uintptr_t>(__cvta_generic_to_shared(stage.b))));
    auto k_update_tensor=make_tensor(k_update_bp,BLogical{});
    auto k_update_b=thr.partition_fragment_B(k_update_tensor);
    load_b_fragment(stage,mma,k_update_b);
#endif

    int last=min(first+kBt,int64_t(eos))-1;
    CUTE_UNROLL
    for(int panel=0;panel<2;++panel) {
#if FLA_RECURRENCE_OVERLAP_AIU
#if FLA_RECURRENCE_REUSE_K_B
      clear(residual);
#if FLA_RECURRENCE_CPASYNC_K
      mma_accumulate_k_token_major_with_b(
          stage,mma,k_update_b,residual);
#else
      mma_accumulate_with_b<AiuATransReader>(stage,mma,k_update_b,residual);
#endif
#else
      clear(residual);
#if FLA_RECURRENCE_CPASYNC_K
      mma_accumulate<AiuATransReader>(stage,mma,residual);
#else
      mma_accumulate<AiuATransReader>(stage,mma,residual);
#endif
#endif
      // The next K producer may overwrite A only after every warp has loaded
      // the current panel's A fragments.
      __syncthreads();
      if(panel==0) {
#if FLA_RECURRENCE_CPASYNC_K
        issue_kg_panel_cpasync(stage,k,first,rows,head,1);
#else
        issue_kg_panel_aiu(stage,k,first,rows,head,1);
#endif
      }
#if !FLA_RECURRENCE_DIRECT_UPDATE
      store_update_panel(stage,residual,coordinates);
      __syncthreads();
#endif
#else
      compute_update_panel(stage,k,first,rows,head,panel);
#endif
#if FLA_RECURRENCE_DIRECT_UPDATE
      update_carried_from_fragment<false>(
          carried,panel,residual,gk,nullptr,chunk,last,head);
#else
#if FLA_RECURRENCE_SHFL_DECAY
      float decay0,decay1;
      load_decay_pair_shfl(gk,last,head,panel,decay0,decay1);
#elif FLA_RECURRENCE_LOCAL_DECAY
      float decay0,decay1;
      load_decay_pair_local(gk,last,head,panel,decay0,decay1);
#endif
      CUTE_UNROLL
      for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
        int value=recurrence_state_v(tid,ordinal);
        int key=recurrence_state_k(tid,ordinal);
        float dot=stage.update[key*kBv+value];
#if FLA_RECURRENCE_SHFL_DECAY || FLA_RECURRENCE_LOCAL_DECAY
        float decay=((ordinal>>2)&1) ? decay1 : decay0;
#else
        float g_last=gk[token_offset(last,head,64*panel+key)];
        float decay=split_exp(g_last);
#endif
        carried[panel][ordinal]=
            __fmaf_rn(carried[panel][ordinal],decay,dot);
      }
#endif
      // Every state owner consumed the update exchange before it is reused.
      __syncthreads();
#if FLA_RECURRENCE_OVERLAP_AIU
      if(panel==0) {
#if FLA_RECURRENCE_CPASYNC_K
        wait_kg_panel_cpasync();
#else
        wait_panel_aiu();
#endif
        __syncthreads();
      }
#endif
    }
  }

  if(final_state) {
    CUTE_UNROLL
    for(int panel=0;panel<2;++panel) {
      CUTE_UNROLL
      for(int ordinal=0;ordinal<kStateOrdinals;++ordinal) {
        int value=recurrence_state_v(tid,ordinal);
        int key=recurrence_state_k(tid,ordinal);
        int64_t offset=state_offset(slot,head,value_base+value,64*panel+key);
        uint16_t rounded=round_bf16(carried[panel][ordinal]);
        if(state_is_fp32) static_cast<float*>(final_state)[offset]=load_bf16(rounded);
        else static_cast<uint16_t*>(final_state)[offset]=rounded;
      }
    }
  }
}
#endif

}  // namespace fla_recurrence

#if !FLA_RECURRENCE_BODY_ONLY
#if FLA_RECURRENCE_PRECOMPUTE_DECAY
// This helper intentionally contains no TensorCell operation.  The exact
// emitted function must remain mma_en=0 so tc01 schedules it over all 64
// generic CUs before the mma_en=1 recurrence consumes the cache.
extern "C" __global__ __launch_bounds__(256,1) __block_size__((256,1,1))
void fla_glm53_recurrence_decay_h8_d128(
    float const* gk,float* decay_cache,int T) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int64_t nt=(int64_t(T)+fla_recurrence::kBt-1)/fla_recurrence::kBt;
  int64_t count=nt*fla_recurrence::kHeads*fla_recurrence::kDim;
  if(index<count) {
    int key=int(index%fla_recurrence::kDim);
    int64_t chunk_head=index/fla_recurrence::kDim;
    int head=int(chunk_head%fla_recurrence::kHeads);
    int64_t chunk=chunk_head/fla_recurrence::kHeads;
    int64_t last=min((chunk+1)*fla_recurrence::kBt,int64_t(T))-1;
    decay_cache[index]=fla_recurrence::reference_exp(
        gk[fla_recurrence::token_offset(last,head,key)]);
  }
#else
  __trap();
#endif
}
#endif

extern "C" __global__
__launch_bounds__(128,FLA_RECURRENCE_MIN_BLOCKS_PER_CU) __block_size__((128,1,1))
void fla_glm53_recurrence_h8_bf16_d128_bt64_nt0(
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,float const* decay_cache,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& stage=*reinterpret_cast<fla_recurrence::Stage*>(dynamic_shared);
#if FLA_RECURRENCE_PIPELINE5
  fla_recurrence::recurrence_body_pipeline5<0>(k,v,w,v_new,gk,decay_cache,h,
      initial_state,final_state,state_is_fp32,state_indices,cu_seqlens,chunk_offsets,T,stage);
#else
  fla_recurrence::recurrence_body<0>(k,v,w,v_new,gk,h,
      initial_state,final_state,state_is_fp32,state_indices,cu_seqlens,chunk_offsets,T,stage);
#endif
#else
  __trap();
#endif
}

extern "C" __global__
__launch_bounds__(128,FLA_RECURRENCE_MIN_BLOCKS_PER_CU) __block_size__((128,1,1))
void fla_glm53_recurrence_h8_bf16_d128_bt64_nt1(
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,float const* decay_cache,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& stage=*reinterpret_cast<fla_recurrence::Stage*>(dynamic_shared);
#if FLA_RECURRENCE_PIPELINE5
  fla_recurrence::recurrence_body_pipeline5<1>(k,v,w,v_new,gk,decay_cache,h,
      initial_state,final_state,state_is_fp32,state_indices,cu_seqlens,chunk_offsets,T,stage);
#else
  fla_recurrence::recurrence_body<1>(k,v,w,v_new,gk,h,
      initial_state,final_state,state_is_fp32,state_indices,cu_seqlens,chunk_offsets,T,stage);
#endif
#else
  __trap();
#endif
}

extern "C" __global__
__launch_bounds__(128,FLA_RECURRENCE_MIN_BLOCKS_PER_CU) __block_size__((128,1,1))
void fla_glm53_recurrence_h8_bf16_d128_bt64_nt2(
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,float const* decay_cache,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& stage=*reinterpret_cast<fla_recurrence::Stage*>(dynamic_shared);
#if FLA_RECURRENCE_PIPELINE5
  fla_recurrence::recurrence_body_pipeline5<2>(k,v,w,v_new,gk,decay_cache,h,
      initial_state,final_state,state_is_fp32,state_indices,cu_seqlens,chunk_offsets,T,stage);
#else
  fla_recurrence::recurrence_body<2>(k,v,w,v_new,gk,h,
      initial_state,final_state,state_is_fp32,state_indices,cu_seqlens,chunk_offsets,T,stage);
#endif
#else
  __trap();
#endif
}

// Each CTA picks the original template by its own sequence; short sequences do not inherit the long-sequence barrier elision.
extern "C" __global__
__launch_bounds__(128,FLA_RECURRENCE_MIN_BLOCKS_PER_CU) __block_size__((128,1,1))
void fla_recurrence_packed(
    uint16_t const* k,uint16_t const* v,uint16_t const* w,uint16_t* v_new,
    float const* gk,float const* decay_cache,uint16_t* h,
    void const* initial_state,void* final_state,int state_is_fp32,
    int32_t const* state_indices,int32_t const* cu_seqlens,
    int64_t const* chunk_offsets,int T) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& stage=*reinterpret_cast<fla_recurrence::Stage*>(dynamic_shared);
  int sequence=blockIdx.z;
  int bos=cu_seqlens[sequence],eos=cu_seqlens[sequence+1];
  if(bos<0 || eos>T || eos<bos) return;
  int nt=(eos-bos+63)/64;
#if FLA_RECURRENCE_PIPELINE5
#define FLA_RUN_BUCKET(bucket) fla_recurrence::recurrence_body_pipeline5<bucket>( \
      k,v,w,v_new,gk,decay_cache,h,initial_state,final_state,state_is_fp32, \
      state_indices,cu_seqlens,chunk_offsets,T,stage)
#else
#define FLA_RUN_BUCKET(bucket) fla_recurrence::recurrence_body<bucket>( \
      k,v,w,v_new,gk,h,initial_state,final_state,state_is_fp32, \
      state_indices,cu_seqlens,chunk_offsets,T,stage)
#endif
  if(nt<=32) { FLA_RUN_BUCKET(0); }
  else if(nt<=128) { FLA_RUN_BUCKET(1); }
  else { FLA_RUN_BUCKET(2); }
#undef FLA_RUN_BUCKET
#else
  __trap();
#endif
}

extern "C" hggcError_t configure_fla_glm53_recurrence_h8_bf16_d128_bt64() {
  hggcError_t status=hggcFuncSetAttribute(
      fla_glm53_recurrence_h8_bf16_d128_bt64_nt0,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      FLA_RECURRENCE_DYNAMIC_SHARED_BYTES);
  if(status!=hggcSuccess) return status;
  status=hggcFuncSetAttribute(
      fla_glm53_recurrence_h8_bf16_d128_bt64_nt1,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      FLA_RECURRENCE_DYNAMIC_SHARED_BYTES);
  if(status!=hggcSuccess) return status;
  status=hggcFuncSetAttribute(
      fla_glm53_recurrence_h8_bf16_d128_bt64_nt2,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      FLA_RECURRENCE_DYNAMIC_SHARED_BYTES);
  if(status!=hggcSuccess) return status;
  return hggcFuncSetAttribute(fla_recurrence_packed,
      hggcFuncAttributeMaxDynamicSharedMemorySize,
      FLA_RECURRENCE_DYNAMIC_SHARED_BYTES);
}

extern "C" hggcError_t launch_fla_glm53_recurrence_h8_bf16_d128_bt64(
    const void* k,const void* v,const void* w,void* v_new,const float* gk,
    float* decay_cache,void* h,const void* initial_state,void* final_state,
    int state_is_fp32,const int32_t* state_indices,
    const int32_t* cu_seqlens,const int64_t* chunk_offsets,int total_tokens,
    int nt_bucket,int n,int heads,hggcStream_t stream) {
  if(nt_bucket < -1 || nt_bucket>2) return hggcErrorInvalidValue;
  if(state_is_fp32<0 || state_is_fp32>1) return hggcErrorInvalidValue;
#if FLA_RECURRENCE_PRECOMPUTE_DECAY
  if(nt_bucket==2) {
    if(decay_cache==nullptr) return hggcErrorInvalidValue;
    int64_t nt=(int64_t(total_tokens)+63)/64;
    int64_t count=nt*FLA_HEADS*128;
    dim3 decay_grid(static_cast<unsigned>((count+255)/256),1,1);
    dim3 decay_block(256,1,1);
    fla_glm53_recurrence_decay_h8_d128<<<decay_grid,decay_block,0,stream>>>(
        gk,decay_cache,total_tokens);
    hggcError_t status=hggcGetLastError();
    if(status!=hggcSuccess) return status;
  }
#endif
#if FLA_RECURRENCE_BV16
  dim3 grid(8,heads,n);
#else
  dim3 grid(4,heads,n);
#endif
  dim3 block(128,1,1);
  auto k16=static_cast<uint16_t const*>(k);
  auto v16=static_cast<uint16_t const*>(v);
  auto w16=static_cast<uint16_t const*>(w);
  auto vn16=static_cast<uint16_t*>(v_new);
  auto h16=static_cast<uint16_t*>(h);
  switch(nt_bucket) {
    case -1:
      fla_recurrence_packed<<<grid,block,
          FLA_RECURRENCE_DYNAMIC_SHARED_BYTES,stream>>>(k16,v16,w16,vn16,gk,
          decay_cache,h16,initial_state,final_state,state_is_fp32,state_indices,
          cu_seqlens,chunk_offsets,total_tokens);
      break;
    case 0:
      fla_glm53_recurrence_h8_bf16_d128_bt64_nt0<<<grid,block,
          FLA_RECURRENCE_DYNAMIC_SHARED_BYTES,stream>>>(k16,v16,w16,vn16,gk,
          decay_cache,h16,initial_state,final_state,state_is_fp32,state_indices,
          cu_seqlens,chunk_offsets,total_tokens);
      break;
    case 1:
      fla_glm53_recurrence_h8_bf16_d128_bt64_nt1<<<grid,block,
          FLA_RECURRENCE_DYNAMIC_SHARED_BYTES,stream>>>(k16,v16,w16,vn16,gk,
          decay_cache,h16,initial_state,final_state,state_is_fp32,state_indices,
          cu_seqlens,chunk_offsets,total_tokens);
      break;
    case 2:
      fla_glm53_recurrence_h8_bf16_d128_bt64_nt2<<<grid,block,
          FLA_RECURRENCE_DYNAMIC_SHARED_BYTES,stream>>>(k16,v16,w16,vn16,gk,
          decay_cache,h16,initial_state,final_state,state_is_fp32,state_indices,
          cu_seqlens,chunk_offsets,total_tokens);
      break;
    default:
      return hggcErrorInvalidValue;
  }
  return hggcGetLastError();
}
#endif  // FLA_RECURRENCE_BODY_ONLY
