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
// Private frozen-domain FLA output candidate for tc01.
//
// Source/CPU contract only: this file is intentionally not wired into the
// production dispatcher. Reference config is 8 warps / 3 stages, mma_en=1,
// and the reference static MMA count is 16.  The candidate emitted count remains unknown
// until a fixed-image build and exact-symbol PASM gate.  Numerical/basis,
// physical correctness, performance, and dispatch are later fail-closed gates.
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

#ifndef FLA_OUTPUT_AIU_H
#define FLA_OUTPUT_AIU_H 0
#endif
#ifndef FLA_OUTPUT_VEC_PUBLISH
#define FLA_OUTPUT_VEC_PUBLISH 0
#endif
#ifndef FLA_OUTPUT_VEC_H
#define FLA_OUTPUT_VEC_H 0
#endif
#ifndef FLA_OUTPUT_NATIVE_H_LAYOUT
#define FLA_OUTPUT_NATIVE_H_LAYOUT 0
#endif
#ifndef FLA_OUTPUT_PACKED_STORE
#define FLA_OUTPUT_PACKED_STORE 0
#endif
#ifndef FLA_OUTPUT_BLOCKED_STORE
#define FLA_OUTPUT_BLOCKED_STORE 0
#endif
#ifndef FLA_OUTPUT_LEADER_AIU_WAIT
#define FLA_OUTPUT_LEADER_AIU_WAIT 0
#endif
#ifndef FLA_OUTPUT_UNROLL_K_PANELS
#define FLA_OUTPUT_UNROLL_K_PANELS 0
#endif
#ifndef FLA_OUTPUT_PAIR_QG_CONVERT
#define FLA_OUTPUT_PAIR_QG_CONVERT 0
#endif
#ifndef FLA_OUTPUT_PRECOMPUTED_POINTERS
#define FLA_OUTPUT_PRECOMPUTED_POINTERS 0
#endif
#ifndef FLA_OUTPUT_SPLIT_QG
#define FLA_OUTPUT_SPLIT_QG 0
#endif
#ifndef FLA_OUTPUT_EXTERNAL_QG
#define FLA_OUTPUT_EXTERNAL_QG 0
#endif
#ifndef FLA_OUTPUT_AIU_QG
#define FLA_OUTPUT_AIU_QG 0
#endif
#ifndef FLA_OUTPUT_PACKED_RECURRENCE_HISTORY
#define FLA_OUTPUT_PACKED_RECURRENCE_HISTORY 0
#endif
#ifndef FLA_OUTPUT_PACKED_DIRECT_READER
#define FLA_OUTPUT_PACKED_DIRECT_READER 0
#endif
#ifndef FLA_OUTPUT_PACKED_SHARED_READER
#define FLA_OUTPUT_PACKED_SHARED_READER 0
#endif
#ifndef FLA_OUTPUT_BFRAGMENT_HISTORY
#define FLA_OUTPUT_BFRAGMENT_HISTORY 0
#endif
#ifndef FLA_OUTPUT_PACKED_VECTOR_READER
#define FLA_OUTPUT_PACKED_VECTOR_READER 0
#endif
#ifndef FLA_OUTPUT_PREFETCH_G_HALVES
#define FLA_OUTPUT_PREFETCH_G_HALVES 0
#endif
#ifndef FLA_OUTPUT_BV128_PAIR_CTA
#define FLA_OUTPUT_BV128_PAIR_CTA 0
#endif
#ifndef FLA_OUTPUT_BV128_PAIR_WARPGROUPS
#define FLA_OUTPUT_BV128_PAIR_WARPGROUPS 0
#endif

// Keep q*scale in FP32 until Qg publication; an intermediate BF16 rounding
// loses accuracy in cancellation-sensitive outputs. Zero is a negative control.
#ifndef FLA_QG_FP32_SCALED_Q
#define FLA_QG_FP32_SCALED_Q 1
#endif
#if FLA_QG_FP32_SCALED_Q != 0 && FLA_QG_FP32_SCALED_Q != 1
#error "FLA_QG_FP32_SCALED_Q must be 0 or 1"
#endif

namespace cute {

// This is a first-compile reader hypothesis.  The matching 8-warp recompute
// reader is useful implementation evidence, but output's qg/H and A/V operand
// bases must be proven by an isolated basis/device gate.
template<bool Trans> struct FlaOutputRawReader {};

template<bool Trans>
struct Copy_Traits<FlaOutputRawReader<Trans>> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32,_128>,Stride<_128,_1>>;
  using DstLayout = Layout<Shape<_32,Shape<_32,_4>>,
                           Stride<_32,Stride<_1,_1024>>>;
  using RefLayout = DstLayout;

  template<class TS,class SL,class TD,class DL>
  CUTE_HOST_DEVICE friend void copy_unpack(
      Copy_Traits const&,Tensor<TS,SL> const& src,Tensor<TD,DL>& dst) {
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
    CUTE_INVALID_CONTROL_PATH("FLA output requires the tc01 device image");
#endif
  }
};

}  // namespace cute

namespace fla_output {

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
using namespace cute;

constexpr int kThreads = 256;
constexpr int kBt = 64;
constexpr int kBk = 64;
constexpr int kBv = 64;
#if FLA_HEADS > 0
constexpr int kHeads = FLA_HEADS;
#else
#define kHeads FLA_DEVICE_HEADS(z)
#endif
constexpr int kDim = 128;
constexpr int kDynamicSharedBytes = 16384;
constexpr bool kRawReaderBasisProved = false;

using Mma=decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{},
    Layout<Shape<_4,_2,_1>,Stride<_2,_1,_8>>{}));
using AReader=Copy_Atom<FlaOutputRawReader<false>,bfloat16_t>;
using BReader=Copy_Atom<FlaOutputRawReader<true>,bfloat16_t>;
using NativeHReader=Copy_Atom<
    PPU0010_TSM_LD_SWZL<bfloat16_t,64,64,false,false,1>,bfloat16_t>;
using ALogical=Layout<Shape<_64,_64>,Stride<_64,_1>>;
using BLogical=Layout<Shape<_64,_64>,Stride<_1,_64>>;

struct alignas(128) Stage {
  uint16_t a[kBt*kBk];
  uint16_t b[kBk*kBv];
};
static_assert(sizeof(Stage) == kDynamicSharedBytes);

#if FLA_OUTPUT_BV128_PAIR_CTA || FLA_OUTPUT_BV128_PAIR_WARPGROUPS
struct alignas(128) PairStage {
  uint16_t a[kBt*kBk];
  uint16_t b0[kBk*kBv];
  uint16_t b1[kBk*kBv];
};
static_assert(sizeof(PairStage)==24576);
#endif

union Bf16Bits {
  uint16_t bits;
  __ppu_bfloat16 value;
};

union Bf16Pack8 {
  uint4 vector;
  uint16_t bits[8];
};

union Bf16Pack2 {
  uint32_t word;
  uint16_t bits[2];
};

CUTE_DEVICE uint16_t round_bf16(float value) {
  Bf16Bits result;
  result.value=__float2bfloat16_rn(value);
  return result.bits;
}

CUTE_DEVICE uint32_t round_bf16x2(float low,float high) {
  __ppu_bfloat162 result=__floats2bfloat162_rn(low,high);
  return *reinterpret_cast<uint32_t*>(&result);
}

CUTE_DEVICE float load_bf16(uint16_t bits) {
  Bf16Bits value;
  value.bits=bits;
  return __bfloat162float(value.value);
}

// This is natural exp(x), expressed in the same log2(e)/exp2 form visible in
// the frozen FLA output ISA, including the deep-negative split.
CUTE_DEVICE float natural_exp(float x) {
  float y=__fmul_rn(x,__uint_as_float(0x3fb8aa3b));
  bool low=y < -126.0f;
  float core=__exp2f(__fmul_rn(y,low ? 0.5f : 1.0f));
  return __fmul_rn(core,low ? core : 1.0f);
}

// The frozen GLM-5.3 TP8 list constrains g to [-5, 0], so the producer never
// enters natural_exp's deep-negative split.  Match the validated Triton Qg
// producer lowering with one ex2.approx while preserving both BF16 roundings.
CUTE_DEVICE float prepare_qg_exp(float x) {
  return __exp2f(__fmul_rn(x,__uint_as_float(0x3fb8aa3b)));
}

FLA_HEAD_OFFSET int64_t token_offset(
    int64_t token,int head,int feature,int width=kDim) {
  return (token*int64_t(kHeads)+head)*int64_t(width)+feature;
}

FLA_HEAD_OFFSET int64_t history_offset(
    int64_t chunk,int head,int value,int key) {
  return ((chunk*int64_t(kHeads)+head)*kDim+value)*kDim+key;
}

CUTE_DEVICE int swizzled(int row,int col) {
  return 64*row+(col^(8*(row%8)));
}

// Exact logical-to-physical permutation of tc01 AIU .swzl for one 64x64
// BF16 cube.  This was exhaustively recovered from all 4096 raw stage slots;
// low three bits are invariant, so every aligned logical vec8 remains a
// physically contiguous uint4 store.
CUTE_DEVICE int aiu_swizzled(int row,int col) {
  int x=64*row+col;
  return (x&7)
      | ((((x>>3)&1)^((x>>8)&1))<<3)
      | ((((x>>5)&1)^((x>>6)&1))<<4)
      | ((((x>>4)&1)^(((x>>5)&1)&((x>>6)&1))^((x>>7)&1))<<5)
      | (((x>>8)&15)<<6)
      | (((x>>4)&3)<<10);
}

CUTE_DEVICE void issue_h_panel_aiu(
    Stage& stage,uint16_t const* h,int64_t history_chunk,int head,
    int value_base,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
#if FLA_OUTPUT_NATIVE_H_LAYOUT
  // Preserve H's [value,key] orientation and consume it through the paired
  // non-transpose raw reader, which supplies the transposed MMA B view.
  using HLoad=PPU0010_AIU_LOAD<
      C<64*64*16>,bfloat16_t,false,true>;
#else
  // Load one contiguous [value,key] tile without hardware swizzle.  Threads
  // remap this linear image into the legacy B-reader layout.
  using HLoad=PPU0010_AIU_LOAD<
      C<64*64*16>,bfloat16_t,false,false>;
#endif
  auto const* input=h+history_offset(
      history_chunk,head,value_base,panel*kBk);
  AiuDesc desc;
  desc.template init<bfloat16_t,false,64,64>(
      reinterpret_cast<uint8_t const*>(input),64,64,
      make_stride(Int<kDim>{},_1{}));
  uint32_t output=static_cast<uint32_t>(__cvta_generic_to_shared(
#if FLA_OUTPUT_NATIVE_H_LAYOUT
      stage.b));
#else
      stage.a));
#endif
  HLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
              reinterpret_cast<void const*>(input),desc,0,0);
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#else
  (void)stage;(void)h;(void)history_chunk;(void)head;
  (void)value_base;(void)panel;
#endif
}

CUTE_DEVICE void wait_output_aiu() {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile("ppu.cp.async.wait_group 0;\n" ::: "memory");
#endif
}

#if FLA_OUTPUT_PACKED_SHARED_READER
CUTE_DEVICE void issue_packed_h_panel_aiu(
    Stage& stage,uint16_t const* h,int64_t chunk,int head,int value_base,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  using HLoad=PPU0010_AIU_LOAD<C<64*64*16>,bfloat16_t,false,false>;
  int tile=2*(value_base/64)+panel;
  auto const* input=h+((chunk*int64_t(kHeads)+head)*4+tile)*4096;
  AiuDesc desc;
  desc.template init<bfloat16_t,false,64,64>(
      reinterpret_cast<uint8_t const*>(input),64,64,make_stride(_64{},_1{}));
  uint32_t output=static_cast<uint32_t>(__cvta_generic_to_shared(stage.b));
  HLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
              reinterpret_cast<void const*>(input),desc,0,0);
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}
#endif

#if FLA_OUTPUT_AIU_QG
CUTE_DEVICE void issue_qg_panel_aiu(
    Stage& stage,uint16_t const* qg,int64_t first,int rows,int head,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  using QgLoad=PPU0010_AIU_LOAD<
      C<64*64*16>,bfloat16_t,false,true>;
  auto const* input=qg+token_offset(first,head,panel*kBk);
  AiuDesc desc;
  desc.template init<bfloat16_t,false,64,64>(
      reinterpret_cast<uint8_t const*>(input),rows,64,
      make_stride(FLA_CUTE_INT(kHeads*kDim),_1{}));
  uint32_t output=static_cast<uint32_t>(
      __cvta_generic_to_shared(stage.a));
  QgLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
               reinterpret_cast<void const*>(input),desc,0,0);
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#else
  (void)stage;(void)qg;(void)first;(void)rows;(void)head;(void)panel;
#endif
}
#endif

#if FLA_OUTPUT_BV128_PAIR_CTA || FLA_OUTPUT_BV128_PAIR_WARPGROUPS
CUTE_DEVICE void issue_pair_h_panel_aiu(
    uint16_t* destination,uint16_t const* h,int64_t history_chunk,int head,
    int value_base,int panel) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  using HLoad=PPU0010_AIU_LOAD<
      C<64*64*16>,bfloat16_t,false,true>;
  auto const* input=h+history_offset(
      history_chunk,head,value_base,panel*kBk);
  AiuDesc desc;
  desc.template init<bfloat16_t,false,64,64>(
      reinterpret_cast<uint8_t const*>(input),64,64,
      make_stride(Int<kDim>{},_1{}));
  uint32_t output=static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  HLoad::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(output)),
              reinterpret_cast<void const*>(input),desc,0,0);
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#else
  (void)destination;(void)h;(void)history_chunk;(void)head;
  (void)value_base;(void)panel;
#endif
}
#endif

CUTE_DEVICE void publish_h_panel_from_linear(Stage& stage) {
  for(int linear=threadIdx.x;linear<kBk*kBv;linear+=kThreads) {
    int value=linear/kBk,key=linear%kBk;
    stage.b[swizzled(key,value)]=stage.a[linear];
  }
}

template<class Pointer>
CUTE_DEVICE auto b_source_slice(Pointer pointer,int tid,int fragment,int q) {
  int warp=tid/32,lane=tid%32;
  auto iterator=make_ptr_inttuple_mix_iter(
      make_coord(16*q+lane%8+8*((lane>>4)&1),
                 16*(warp%2)+32*fragment+(lane&8)),pointer);
  return make_tensor(iterator,make_layout(make_shape(_8{})));
}

template<bool NativeLayout,class BCopyAtom,class Accumulator>
CUTE_DEVICE void mma_accumulate_with_b(
    Stage& stage,Mma const& mma,Accumulator& accumulator) {
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
  // Qg arrives in the AIU hardware-swizzled image, whereas the later A*V
  // phase is still published with the legacy software-swizzled layout.
  using PhaseAReader=conditional_t<NativeLayout && bool(FLA_OUTPUT_AIU_QG),
                                  NativeHReader,AReader>;
  auto copy_a=make_tiled_copy_A(PhaseAReader{},mma);
  auto copy_b=make_tiled_copy_B(BCopyAtom{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto s_b=copy_b.get_thread_slice(tid).partition_S(make_mix_tensor_like(b));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  auto d_b=copy_b.get_thread_slice(tid).retile_D(r_b);
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    if constexpr (NativeLayout) {
      copy(copy_b,s_b(_,_,q),d_b(_,_,q));
    } else {
      CUTE_UNROLL
      for(int fragment=0;fragment<2;++fragment) {
        auto src=b_source_slice(bp,tid,fragment,q);
        auto dst=d_b(_,fragment,q);
        BCopyAtom{}.call(src,dst);
      }
    }
    cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
  }
}

template<class Accumulator>
CUTE_DEVICE void mma_accumulate(
    Stage& stage,Mma const& mma,Accumulator& accumulator) {
  mma_accumulate_with_b<false,BReader>(stage,mma,accumulator);
}

template<class Accumulator>
CUTE_DEVICE void mma_accumulate_native_h(
    Stage& stage,Mma const& mma,Accumulator& accumulator) {
  mma_accumulate_with_b<true,NativeHReader>(stage,mma,accumulator);
}

// Invert recurrence's (thread, register ordinal) history publication.  B is
// logically [value,key]; deriving coordinates from the MMA partition avoids
// assuming the recurrence C fragment and output B fragment share an ABI.
CUTE_HOST_DEVICE int packed_history_element(int value,int key) {
  int tid=32*(key/16)+4*(key%8)+value%4;
  int ordinal=(value%16)/4+4*((key%16)/8);
  return (value/16)*1024+8*tid+ordinal;
}

#if FLA_OUTPUT_PACKED_DIRECT_READER
template<class Accumulator>
CUTE_DEVICE void mma_accumulate_packed_h(
    Stage& stage,Mma const& mma,Accumulator& accumulator,
    uint16_t const* h,int64_t chunk,int head,int value_base,int panel) {
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(stage.a))));
  auto a=make_tensor(ap,ALogical{});
  auto thr=mma.get_thread_slice(threadIdx.x);
  auto r_a=thr.partition_fragment_A(a);
  auto coords=thr.partition_B(make_identity_tensor(make_shape(_64{},_64{})));
  auto r_b=make_tensor<bfloat16_t>(shape(coords));
  using PackedAReader=conditional_t<bool(FLA_OUTPUT_AIU_QG),NativeHReader,AReader>;
  auto copy_a=make_tiled_copy_A(PackedAReader{},mma);
  auto s_a=copy_a.get_thread_slice(threadIdx.x).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(threadIdx.x).retile_D(r_a);
#if FLA_OUTPUT_PACKED_SHARED_READER
  auto const* input=reinterpret_cast<bfloat16_t const*>(stage.b);
#else
  int tile=2*(value_base/64)+panel;
  auto const* input=reinterpret_cast<bfloat16_t const*>(
      h+((chunk*int64_t(kHeads)+head)*4+tile)*4096);
#endif
  CUTE_UNROLL
  for(int q=0;q<4;++q) {
    auto fragment=r_b(_,_,q);
#if FLA_OUTPUT_PACKED_VECTOR_READER
    CUTE_STATIC_ASSERT_V(size<0>(fragment)==_8{});
    CUTE_STATIC_ASSERT_V(size<1>(fragment)==_2{});
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
#if FLA_OUTPUT_BFRAGMENT_HISTORY
    // Paired only with B-fragment-ready history publication. Each lane
    // reads its final four MMA B words; no cross-lane exchange is needed.
    CUTE_UNROLL
    for(int n=0;n<2;++n) {
      int quarter=(warp&1)+2*n;
      uint4 packed=*reinterpret_cast<uint4 const*>(
          input+1024*quarter+256*q+8*lane);
      auto destination=fragment(_,n);
      auto* words=reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(destination.data()));
      words[0]=packed.x; words[1]=packed.y;
      words[2]=packed.z; words[3]=packed.w;
    }
#else
    int producer_lane=8*(lane&3)+((lane>>2)&3)+4*(lane>>4);
    uint32_t selector=(lane&16) ? 0x3276u : 0x5410u;
    CUTE_UNROLL
    for(int n=0;n<2;++n) {
      int quarter=(warp&1)+2*n;
      // All 32 producer records form one contiguous 512-byte span. Each
      // lane loads one aligned record; XOR-16 exchanges its paired key.
      uint4 packed=*reinterpret_cast<uint4 const*>(
          input+1024*quarter+256*q+8*producer_lane);
      uint32_t x=__shfl_xor_sync(0xffffffffu,packed.x,16,32);
      uint32_t y=__shfl_xor_sync(0xffffffffu,packed.y,16,32);
      uint32_t z=__shfl_xor_sync(0xffffffffu,packed.z,16,32);
      uint32_t w=__shfl_xor_sync(0xffffffffu,packed.w,16,32);
      auto destination=fragment(_,n);
      auto* words=reinterpret_cast<uint32_t*>(cute::raw_pointer_cast(destination.data()));
      // tc01 B word order is producer word 0,2,1,3. Lower lanes take
      // local.low / partner.low; upper lanes take partner.high / local.high.
      words[0]=__byte_perm(packed.x,x,selector);
      words[1]=__byte_perm(packed.z,z,selector);
      words[2]=__byte_perm(packed.y,y,selector);
      words[3]=__byte_perm(packed.w,w,selector);
    }
#endif
#else
    auto positions=coords(_,_,q);
    CUTE_UNROLL
    for(int i=0;i<size(fragment);++i) {
      auto coord=positions(i);
      fragment(i)=input[packed_history_element(int(get<0>(coord)),int(get<1>(coord)))];
    }
#endif
    copy(copy_a,s_a(_,_,q),d_a(_,_,q));
    cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
  }
}
#endif

#if FLA_OUTPUT_BV128_PAIR_CTA || FLA_OUTPUT_BV128_PAIR_WARPGROUPS
template<bool NativeLayout,class Accumulator>
CUTE_DEVICE void mma_accumulate_pair_tid(
    uint16_t* a_storage,uint16_t* b_storage,Mma const& mma,
    Accumulator& accumulator,int tid) {
  auto ap=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(a_storage))));
  auto bp=make_smem_ptr(reinterpret_cast<bfloat16_t*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(b_storage))));
  auto a=make_tensor(ap,ALogical{});
  auto b=make_tensor(bp,BLogical{});
  auto thr=mma.get_thread_slice(tid);
  auto r_a=thr.partition_fragment_A(a);
  auto r_b=thr.partition_fragment_B(b);
  auto copy_a=make_tiled_copy_A(AReader{},mma);
  auto s_a=copy_a.get_thread_slice(tid).partition_S(make_mix_tensor_like(a));
  auto d_a=copy_a.get_thread_slice(tid).retile_D(r_a);
  if constexpr (NativeLayout) {
    auto copy_b=make_tiled_copy_B(NativeHReader{},mma);
    auto s_b=copy_b.get_thread_slice(tid).partition_S(make_mix_tensor_like(b));
    auto d_b=copy_b.get_thread_slice(tid).retile_D(r_b);
    CUTE_UNROLL
    for(int q=0;q<4;++q) {
      copy(copy_a,s_a(_,_,q),d_a(_,_,q));
      copy(copy_b,s_b(_,_,q),d_b(_,_,q));
      cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
    }
  } else {
    auto copy_b=make_tiled_copy_B(BReader{},mma);
    auto d_b=copy_b.get_thread_slice(tid).retile_D(r_b);
    CUTE_UNROLL
    for(int q=0;q<4;++q) {
      copy(copy_a,s_a(_,_,q),d_a(_,_,q));
      CUTE_UNROLL
      for(int fragment=0;fragment<2;++fragment) {
        auto src=b_source_slice(bp,tid,fragment,q);
        auto dst=d_b(_,fragment,q);
        BReader{}.call(src,dst);
      }
      cute::gemm(mma,r_a(_,_,q),r_b(_,_,q),accumulator);
    }
  }
}

#if FLA_OUTPUT_BV128_PAIR_CTA
template<bool NativeLayout,class Accumulator>
CUTE_DEVICE void mma_accumulate_pair(
    uint16_t* a_storage,uint16_t* b_storage,Mma const& mma,
    Accumulator& accumulator) {
  mma_accumulate_pair_tid<NativeLayout>(
      a_storage,b_storage,mma,accumulator,threadIdx.x);
}
#endif

#if FLA_OUTPUT_BV128_PAIR_WARPGROUPS
template<bool NativeLayout,class Accumulator>
CUTE_DEVICE void mma_accumulate_warpgroup(
    uint16_t* a_storage,uint16_t* b_storage,Mma const& mma,
    Accumulator& accumulator,int local_tid) {
  mma_accumulate_pair_tid<NativeLayout>(
      a_storage,b_storage,mma,accumulator,local_tid);
}
#endif
#endif

CUTE_DEVICE void publish_qg_h_panel(
    Stage& stage,uint16_t const* q,float const* g,uint16_t const* h,
    int64_t first,int rows,int head,int value_base,int panel,
    int64_t history_chunk,float scale) {
  for(int linear=threadIdx.x;linear<kBt*kBk;linear+=kThreads) {
    int row=linear/kBk,col=linear%kBk;
    // A is [token,key], while B is H^T=[key,value].  The same physical
    // (row,col) therefore names different logical source axes in each tile.
    int q_key=panel*kBk+col;
    int h_key=panel*kBk+row;
    uint16_t qg=uint16_t(0);
    if(row<rows) {
      float q_value=load_bf16(q[token_offset(first+row,head,q_key)]);
#if FLA_QG_FP32_SCALED_Q
      float q_scaled=__fmul_rn(q_value,scale);
      float g_value=g[token_offset(first+row,head,q_key)];
      qg=round_bf16(__fmul_rn(q_scaled,natural_exp(g_value)));
#else
      uint16_t q_scaled=round_bf16(__fmul_rn(q_value,scale));
      float g_value=g[token_offset(first+row,head,q_key)];
      qg=round_bf16(__fmul_rn(load_bf16(q_scaled),natural_exp(g_value)));
#endif
    }
    stage.a[swizzled(row,col)]=qg;
#if !FLA_OUTPUT_AIU_H
    int value=value_base+col;
    stage.b[swizzled(row,col)]=
        h[history_offset(history_chunk,head,value,h_key)];
#endif
  }
}

CUTE_DEVICE void publish_qg_panel_vec(
    Stage& stage,uint16_t const* q,float const* g,
    int64_t first,int rows,int head,int panel,float scale) {
  int col=8*(threadIdx.x%8);
#if FLA_OUTPUT_PRECOMPUTED_POINTERS
  int row0=threadIdx.x/8;
  int64_t offset0=((first+row0)*kHeads+head)*int64_t(kDim)+panel*kBk+col;
  auto const* q0=q+offset0;
  auto const* g0=g+offset0;
  const int64_t qg_half_stride=32LL*kHeads*kDim;
#endif
#if FLA_OUTPUT_PREFETCH_G_HALVES
  static_assert(FLA_OUTPUT_PRECOMPUTED_POINTERS);
  float prefetched_gates[2][8]{};
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=row0+half*32;
    if(row<rows) {
      auto const* gate_pointer=g0+half*qg_half_stride;
      float4 low=*reinterpret_cast<float4 const*>(gate_pointer);
      float4 high=*reinterpret_cast<float4 const*>(gate_pointer+4);
      prefetched_gates[half][0]=low.x;
      prefetched_gates[half][1]=low.y;
      prefetched_gates[half][2]=low.z;
      prefetched_gates[half][3]=low.w;
      prefetched_gates[half][4]=high.x;
      prefetched_gates[half][5]=high.y;
      prefetched_gates[half][6]=high.z;
      prefetched_gates[half][7]=high.w;
    }
  }
#endif
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=threadIdx.x/8+half*32;
    Bf16Pack8 output{};
    if(row<rows) {
#if FLA_OUTPUT_PRECOMPUTED_POINTERS
      auto const* q_pointer=q0+half*qg_half_stride;
#if !FLA_OUTPUT_PREFETCH_G_HALVES
      auto const* g_pointer=g0+half*qg_half_stride;
#endif
#else
      auto const* q_pointer=q+token_offset(first+row,head,panel*kBk+col);
      auto const* g_pointer=g+token_offset(first+row,head,panel*kBk+col);
#endif
      Bf16Pack8 q_values;
      q_values.vector=*reinterpret_cast<uint4 const*>(q_pointer);
#if FLA_OUTPUT_PREFETCH_G_HALVES
      float const* gates=prefetched_gates[half];
#else
      float4 g_low=*reinterpret_cast<float4 const*>(g_pointer);
      float4 g_high=*reinterpret_cast<float4 const*>(g_pointer+4);
      float gates_local[8]={g_low.x,g_low.y,g_low.z,g_low.w,
                            g_high.x,g_high.y,g_high.z,g_high.w};
      float const* gates=gates_local;
#endif
#if FLA_OUTPUT_PAIR_QG_CONVERT
      CUTE_UNROLL
      for(int pair=0;pair<4;++pair) {
        int low=2*pair,high=low+1;
#if FLA_QG_FP32_SCALED_Q
        float q_low=__fmul_rn(load_bf16(q_values.bits[low]),scale);
        float q_high=__fmul_rn(load_bf16(q_values.bits[high]),scale);
        Bf16Pack2 qg;
        qg.word=round_bf16x2(
            __fmul_rn(q_low,natural_exp(gates[low])),
            __fmul_rn(q_high,natural_exp(gates[high])));
#else
        Bf16Pack2 q_scaled;
        q_scaled.word=round_bf16x2(
            __fmul_rn(load_bf16(q_values.bits[low]),scale),
            __fmul_rn(load_bf16(q_values.bits[high]),scale));
        Bf16Pack2 qg;
        qg.word=round_bf16x2(
            __fmul_rn(load_bf16(q_scaled.bits[0]),natural_exp(gates[low])),
            __fmul_rn(load_bf16(q_scaled.bits[1]),natural_exp(gates[high])));
#endif
        output.bits[low]=qg.bits[0];
        output.bits[high]=qg.bits[1];
      }
#else
      CUTE_UNROLL
      for(int item=0;item<8;++item) {
#if FLA_QG_FP32_SCALED_Q
        float q_scaled=__fmul_rn(load_bf16(q_values.bits[item]),scale);
        output.bits[item]=round_bf16(__fmul_rn(
            q_scaled,natural_exp(gates[item])));
#else
        uint16_t q_scaled=round_bf16(
            __fmul_rn(load_bf16(q_values.bits[item]),scale));
        output.bits[item]=round_bf16(__fmul_rn(
            load_bf16(q_scaled),natural_exp(gates[item])));
#endif
      }
#endif
    }
    *reinterpret_cast<uint4*>(stage.a+swizzled(row,col))=output.vector;
  }
}

CUTE_DEVICE void publish_precomputed_qg_panel_vec(
    Stage& stage,uint16_t const* qg,int64_t first,int rows,int head,int panel) {
  int col=8*(threadIdx.x%8);
  int row0=threadIdx.x/8;
  int64_t offset0=((first+row0)*kHeads+head)*int64_t(kDim)+panel*kBk+col;
  auto const* qg0=qg+offset0;
  const int64_t half_stride=32LL*kHeads*kDim;
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=row0+half*32;
    Bf16Pack8 values{};
    if(row<rows)
      values.vector=*reinterpret_cast<uint4 const*>(qg0+half*half_stride);
    *reinterpret_cast<uint4*>(stage.a+swizzled(row,col))=values.vector;
  }
}

CUTE_DEVICE void prepare_qg_vec8(
    uint16_t const* q,float const* g,uint16_t* qg,int64_t first,int rows,
    int head,float scale) {
  int col=8*(threadIdx.x%16);
  int row0=threadIdx.x/16;
  CUTE_UNROLL
  for(int quarter=0;quarter<4;++quarter) {
    int row=row0+16*quarter;
    if(row<rows) {
      int64_t offset=token_offset(first+row,head,col);
      Bf16Pack8 q_values,output;
      q_values.vector=*reinterpret_cast<uint4 const*>(q+offset);
      auto const* gate_pointer=g+offset;
      float4 gate_low=*reinterpret_cast<float4 const*>(gate_pointer);
      float4 gate_high=*reinterpret_cast<float4 const*>(gate_pointer+4);
      float gates[8]={gate_low.x,gate_low.y,gate_low.z,gate_low.w,
                      gate_high.x,gate_high.y,gate_high.z,gate_high.w};
      CUTE_UNROLL
      for(int pair=0;pair<4;++pair) {
        int low=2*pair,high=low+1;
#if FLA_QG_FP32_SCALED_Q
        float q_low=__fmul_rn(load_bf16(q_values.bits[low]),scale);
        float q_high=__fmul_rn(load_bf16(q_values.bits[high]),scale);
        Bf16Pack2 qg_pair;
        qg_pair.word=round_bf16x2(
            __fmul_rn(q_low,prepare_qg_exp(gates[low])),
            __fmul_rn(q_high,prepare_qg_exp(gates[high])));
#else
        Bf16Pack2 q_scaled;
        q_scaled.word=round_bf16x2(
            __fmul_rn(load_bf16(q_values.bits[low]),scale),
            __fmul_rn(load_bf16(q_values.bits[high]),scale));
        Bf16Pack2 qg_pair;
        qg_pair.word=round_bf16x2(
            __fmul_rn(load_bf16(q_scaled.bits[0]),prepare_qg_exp(gates[low])),
            __fmul_rn(load_bf16(q_scaled.bits[1]),prepare_qg_exp(gates[high])));
#endif
        output.bits[low]=qg_pair.bits[0];
        output.bits[high]=qg_pair.bits[1];
      }
      *reinterpret_cast<uint4*>(qg+offset)=output.vector;
    }
  }
}

CUTE_DEVICE void publish_h_panel_vec(
    Stage& stage,uint16_t const* h,int64_t history_chunk,int head,
    int value_base,int panel) {
  int col=8*(threadIdx.x%8);
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=threadIdx.x/8+half*32;
    auto const* input=h+history_offset(
        history_chunk,head,value_base+row,panel*kBk+col);
    Bf16Pack8 values;
    values.vector=*reinterpret_cast<uint4 const*>(input);
    *reinterpret_cast<uint4*>(stage.b+aiu_swizzled(row,col))=values.vector;
  }
}

CUTE_DEVICE void publish_packed_recurrence_h_panel(
    Stage& stage,uint16_t const* h,int64_t history_chunk,int head,
    int value_base,int panel) {
  int value_tile=value_base/64;
  int tile=2*value_tile+panel;
  auto const* input=h+((history_chunk*int64_t(kHeads)+head)*4+tile)*4096;
  CUTE_UNROLL
  for(int pass=0;pass<2;++pass) {
    int pack_index=threadIdx.x+256*pass;
    int quarter=pack_index/128;
    int recurrence_tid=pack_index%128;
    Bf16Pack8 packed;
    packed.vector=*reinterpret_cast<uint4 const*>(
        input+quarter*1024+8*recurrence_tid);
    int lane=recurrence_tid%32;
    int warp=recurrence_tid/32;
    CUTE_UNROLL
    for(int ordinal=0;ordinal<8;++ordinal) {
      int key=(lane>>2)+8*((ordinal>>2)&1)+16*warp;
      int value=16*quarter+(lane&3)+4*(ordinal&1)+8*((ordinal>>1)&1);
      stage.b[aiu_swizzled(key,value)]=packed.bits[ordinal];
    }
  }
}

CUTE_DEVICE void publish_lower_A_and_v(
    Stage& stage,uint16_t const* A,uint16_t const* v,
    int64_t first,int rows,int head,int value_base) {
  for(int linear=threadIdx.x;linear<kBt*kBv;linear+=kThreads) {
    int row=linear/kBv,col=linear%kBv;
    bool valid_row=row<rows;
    bool in_lower_triangle=valid_row && col<=row;
    stage.a[swizzled(row,col)]=in_lower_triangle
        ? round_bf16(load_bf16(A[token_offset(first+row,head,col,kBt)]))
        : uint16_t(0);
    stage.b[swizzled(row,col)]=valid_row
        ? v[token_offset(first+row,head,value_base+col)] : uint16_t(0);
  }
}

CUTE_DEVICE void publish_lower_A_and_v_vec(
    Stage& stage,uint16_t const* A,uint16_t const* v,
    int64_t first,int rows,int head,int value_base) {
  int col=8*(threadIdx.x%8);
#if FLA_OUTPUT_PRECOMPUTED_POINTERS
  int row0=threadIdx.x/8;
  auto const* A0=A+((first+row0)*kHeads+head)*int64_t(kBt)+col;
  auto const* v0=v+((first+row0)*kHeads+head)*int64_t(kDim)+value_base+col;
  const int64_t A_half_stride=32LL*kHeads*kBt;
  const int64_t v_half_stride=32LL*kHeads*kDim;
#endif
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=threadIdx.x/8+half*32;
    Bf16Pack8 a_values{},v_values{};
    if(row<rows) {
#if FLA_OUTPUT_PRECOMPUTED_POINTERS
      a_values.vector=*reinterpret_cast<uint4 const*>(A0+half*A_half_stride);
      v_values.vector=*reinterpret_cast<uint4 const*>(v0+half*v_half_stride);
#else
      a_values.vector=*reinterpret_cast<uint4 const*>(
          A+token_offset(first+row,head,col,kBt));
      v_values.vector=*reinterpret_cast<uint4 const*>(
          v+token_offset(first+row,head,value_base+col));
#endif
      CUTE_UNROLL
      for(int item=0;item<8;++item)
        if(col+item>row) a_values.bits[item]=uint16_t(0);
    }
    *reinterpret_cast<uint4*>(stage.a+swizzled(row,col))=a_values.vector;
    *reinterpret_cast<uint4*>(stage.b+swizzled(row,col))=v_values.vector;
  }
}

#if FLA_OUTPUT_BV128_PAIR_CTA
CUTE_DEVICE void publish_pair_A_and_v(
    PairStage& pair,uint16_t const* A,uint16_t const* v,
    int64_t first,int rows,int head) {
  int col=8*(threadIdx.x%8);
  int row0=threadIdx.x/8;
  auto const* A0=A+((first+row0)*kHeads+head)*int64_t(kBt)+col;
  auto const* v0=v+((first+row0)*kHeads+head)*int64_t(kDim)+col;
  const int64_t A_half_stride=32LL*kHeads*kBt;
  const int64_t v_half_stride=32LL*kHeads*kDim;
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=row0+half*32;
    Bf16Pack8 a_values{},v_values0{},v_values1{};
    if(row<rows) {
      a_values.vector=*reinterpret_cast<uint4 const*>(A0+half*A_half_stride);
      v_values0.vector=*reinterpret_cast<uint4 const*>(v0+half*v_half_stride);
      v_values1.vector=*reinterpret_cast<uint4 const*>(
          v0+half*v_half_stride+kBv);
      CUTE_UNROLL
      for(int item=0;item<8;++item)
        if(col+item>row) a_values.bits[item]=uint16_t(0);
    }
    int destination=swizzled(row,col);
    *reinterpret_cast<uint4*>(pair.a+destination)=a_values.vector;
    *reinterpret_cast<uint4*>(pair.b0+destination)=v_values0.vector;
    *reinterpret_cast<uint4*>(pair.b1+destination)=v_values1.vector;
  }
}
#endif

#if FLA_OUTPUT_BV128_PAIR_WARPGROUPS
CUTE_DEVICE void publish_warpgroup_A_and_v(
    PairStage& pair,uint16_t const* A,uint16_t const* v,
    int64_t first,int rows,int head,int group,int local_tid) {
  int col=8*(local_tid%8);
  int row0=local_tid/8;
  auto const* A0=A+((first+row0)*kHeads+head)*int64_t(kBt)+col;
  auto const* v0=v+((first+row0)*kHeads+head)*int64_t(kDim)
      +group*kBv+col;
  const int64_t A_half_stride=32LL*kHeads*kBt;
  const int64_t v_half_stride=32LL*kHeads*kDim;
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=row0+half*32;
    Bf16Pack8 a_values{},v_values{};
    if(row<rows) {
      v_values.vector=*reinterpret_cast<uint4 const*>(v0+half*v_half_stride);
      if(group==0) {
        a_values.vector=*reinterpret_cast<uint4 const*>(A0+half*A_half_stride);
        CUTE_UNROLL
        for(int item=0;item<8;++item)
          if(col+item>row) a_values.bits[item]=uint16_t(0);
      }
    }
    int destination=swizzled(row,col);
    if(group==0)
      *reinterpret_cast<uint4*>(pair.a+destination)=a_values.vector;
    uint16_t* b_storage=group==0 ? pair.b0 : pair.b1;
    *reinterpret_cast<uint4*>(b_storage+destination)=v_values.vector;
  }
}
#endif

template<class Accumulator,class Coordinates>
CUTE_DEVICE void store_output(
    Accumulator const& accumulator,Coordinates const& coordinates,uint16_t* output,
    int64_t first,int rows,int head,int value_base) {
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(accumulator));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(accumulator));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(accumulator));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int row=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        if(row<rows && value<kBv)
          output[token_offset(first+row,head,value_base+value)]=
              round_bf16(accumulator(value_index,m,n));
  }
}

template<class Accumulator,class Coordinates>
CUTE_DEVICE void store_packed_output(
    Stage& stage,Accumulator const& accumulator,Coordinates const& coordinates,
    uint16_t* output,int64_t first,int rows,int head,int value_base) {
  __syncthreads();
  CUTE_UNROLL
  for(int m=0;m<int(size<1>(accumulator));++m)
    CUTE_UNROLL
    for(int n=0;n<int(size<2>(accumulator));++n)
      CUTE_UNROLL
      for(int value_index=0;value_index<int(size<0>(accumulator));++value_index) {
        auto coordinate=coordinates(value_index,m,n);
        int row=int(get<0>(coordinate));
        int value=int(get<1>(coordinate));
        if(row<kBt && value<kBv)
          stage.a[kBv*row+value]=round_bf16(accumulator(value_index,m,n));
      }
  __syncthreads();
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int linear=8*threadIdx.x+half*(kBt*kBv/2);
    int row=linear/kBv;
    int value=linear%kBv;
    if(row<rows) {
      uint4 packed=*reinterpret_cast<uint4 const*>(stage.a+linear);
      *reinterpret_cast<uint4*>(output+token_offset(
          first+row,head,value_base+value))=packed;
    }
  }
}

// Convert the MMA fragment to the reference blocked output ownership without
// issuing 16 scattered BF16 global stores per thread.  Each producer first
// packs the register pairs used by one PPU MMA atom into eight TSM words.  A
// blocked-layout consumer then gathers four producer words per output row and
// emits the same two aligned uint4 global stores as frozen Triton.
template<class Accumulator>
CUTE_DEVICE void store_blocked_output(
    Stage& stage,Accumulator const& accumulator,uint16_t* output,
    int64_t first,int rows,int head,int value_base) {
  static_assert(size<0>(Accumulator{})==8);
  static_assert(size<1>(Accumulator{})==1);
  static_assert(size<2>(Accumulator{})==2);
  __syncthreads();
  auto* words=reinterpret_cast<uint32_t*>(stage.a);
  CUTE_UNROLL
  for(int n=0;n<2;++n) {
    CUTE_UNROLL
    for(int pair=0;pair<4;++pair) {
      Bf16Pack2 packed;
      packed.bits[0]=round_bf16(accumulator(2*pair,0,n));
      packed.bits[1]=round_bf16(accumulator(2*pair+1,0,n));
      words[8*threadIdx.x+4*n+pair]=packed.word;
    }
  }
  __syncthreads();

  int lane=threadIdx.x%32;
  int consumer_warp=threadIdx.x/32;
  int col=8*(lane%8);
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=4*consumer_warp+lane/8+32*half;
    int producer_warp=2*(row/16)+(col%32)/16;
    int producer_lane=4*(row%8);
    int producer_tid=32*producer_warp+producer_lane;
    int n=col/32;
    int pair=(col%16)/8+2*((row%16)/8);
    uint32_t gathered[4];
    CUTE_UNROLL
    for(int item=0;item<4;++item)
      gathered[item]=words[8*(producer_tid+item)+4*n+pair];
    Bf16Pack8 packed;
    CUTE_UNROLL
    for(int item=0;item<4;++item) {
      packed.bits[item]=uint16_t(gathered[item]);
      packed.bits[item+4]=uint16_t(gathered[item]>>16);
    }
    if(row<rows)
      *reinterpret_cast<uint4*>(output+token_offset(
          first+row,head,value_base+col))=packed.vector;
  }
}

#if FLA_OUTPUT_BV128_PAIR_WARPGROUPS
template<class Accumulator>
CUTE_DEVICE void store_blocked_output_warpgroup(
    uint16_t* scratch,Accumulator const& accumulator,uint16_t* output,
    int64_t first,int rows,int head,int value_base,int local_tid) {
  static_assert(size<0>(Accumulator{})==8);
  static_assert(size<1>(Accumulator{})==1);
  static_assert(size<2>(Accumulator{})==2);
  __syncthreads();
  auto* words=reinterpret_cast<uint32_t*>(scratch);
  CUTE_UNROLL
  for(int n=0;n<2;++n) {
    CUTE_UNROLL
    for(int pair=0;pair<4;++pair) {
      Bf16Pack2 packed;
      packed.bits[0]=round_bf16(accumulator(2*pair,0,n));
      packed.bits[1]=round_bf16(accumulator(2*pair+1,0,n));
      words[8*local_tid+4*n+pair]=packed.word;
    }
  }
  __syncthreads();

  int lane=local_tid%32;
  int consumer_warp=local_tid/32;
  int col=8*(lane%8);
  CUTE_UNROLL
  for(int half=0;half<2;++half) {
    int row=4*consumer_warp+lane/8+32*half;
    int producer_warp=2*(row/16)+(col%32)/16;
    int producer_lane=4*(row%8);
    int producer_tid=32*producer_warp+producer_lane;
    int n=col/32;
    int pair=(col%16)/8+2*((row%16)/8);
    uint32_t gathered[4];
    CUTE_UNROLL
    for(int item=0;item<4;++item)
      gathered[item]=words[8*(producer_tid+item)+4*n+pair];
    Bf16Pack8 packed;
    CUTE_UNROLL
    for(int item=0;item<4;++item) {
      packed.bits[item]=uint16_t(gathered[item]);
      packed.bits[item+4]=uint16_t(gathered[item]>>16);
    }
    if(row<rows)
      *reinterpret_cast<uint4*>(output+token_offset(
          first+row,head,value_base+col))=packed.vector;
  }
}
#endif
#endif

}  // namespace fla_output

#if FLA_OUTPUT_SPLIT_QG
extern "C" __global__ __launch_bounds__(256,1) __block_size__((256,1,1))
void fla_glm53_prepare_qg_h8_bf16_d128_bt64(
    uint16_t const* q,float const* g,uint16_t* qg,int32_t const* cu,
    int32_t const* indices,float scale,int total_tokens,int n) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  using namespace fla_output;
  int history_chunk=blockIdx.x;
  int head=blockIdx.z;
  int sequence=indices[2*history_chunk];
  int chunk=indices[2*history_chunk+1];
  if(sequence<0 || sequence>=n || chunk<0) return;
  int bos=cu[sequence],eos=cu[sequence+1];
  int64_t first=int64_t(bos)+int64_t(chunk)*kBt;
  if(bos<0 || eos>total_tokens || first<bos || first>=eos) return;
  prepare_qg_vec8(q,g,qg,first,min(kBt,eos-int(first)),head,scale);
#else
  __trap();
#endif
}
#endif

extern "C" __global__ __launch_bounds__(256,1) __block_size__((256,1,1))
void fla_glm53_output_h8_bf16_d128_bt64(
    uint16_t const* q,uint16_t const* v,float const* g,uint16_t const* h,
    uint16_t* output,uint16_t const* A,int32_t const* cu,
    int32_t const* indices,float scale,int total_tokens,int n) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  using namespace fla_output;
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& stage=*reinterpret_cast<Stage*>(dynamic_shared);
  int value_base=blockIdx.x*kBv;
  int history_chunk=blockIdx.y;
  int head=blockIdx.z;
  int sequence=indices[2*history_chunk];
  int chunk=indices[2*history_chunk+1];
  if(sequence<0 || sequence>=n || chunk<0) return;
  int bos=cu[sequence],eos=cu[sequence+1];
  int64_t first=int64_t(cu[sequence])+int64_t(chunk)*kBt;
  if(bos<0 || eos>total_tokens || first<bos || first>=eos) return;
  int rows=min(kBt,eos-int(first));

  Mma mma;
  auto thr=mma.get_thread_slice(threadIdx.x);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},_64{})));
  auto accumulator=Mma::make_fragment_C(coordinates);
  clear(accumulator);

  // The frozen Triton range has two compile-time BK64 iterations.  Keep a
  // switchable runtime form so the scheduler effect can be isolated on tc01.
#if FLA_OUTPUT_UNROLL_K_PANELS
  CUTE_UNROLL
#else
  #pragma unroll 1
#endif
  for(int panel=0;panel<2;++panel) {
    if(panel!=0) __syncthreads();
#if FLA_OUTPUT_PACKED_SHARED_READER
    if(threadIdx.x==0)
      issue_packed_h_panel_aiu(stage,h,history_chunk,head,value_base,panel);
#endif
#if FLA_OUTPUT_AIU_H && !FLA_OUTPUT_PACKED_RECURRENCE_HISTORY
    if(threadIdx.x==0) issue_h_panel_aiu(
        stage,h,history_chunk,head,value_base,panel);
#endif
#if FLA_OUTPUT_SPLIT_QG
#if FLA_OUTPUT_AIU_QG
    if(threadIdx.x==0)
      issue_qg_panel_aiu(stage,q,first,rows,head,panel);
#else
    publish_precomputed_qg_panel_vec(stage,q,first,rows,head,panel);
#endif
#elif FLA_OUTPUT_VEC_PUBLISH && FLA_OUTPUT_NATIVE_H_LAYOUT
    publish_qg_panel_vec(stage,q,g,first,rows,head,panel,scale);
#endif
#if FLA_OUTPUT_VEC_H && FLA_OUTPUT_NATIVE_H_LAYOUT && !FLA_OUTPUT_AIU_H && \
    !FLA_OUTPUT_PACKED_RECURRENCE_HISTORY
    publish_h_panel_vec(
        stage,h,history_chunk,head,value_base,panel);
#endif
#if FLA_OUTPUT_PACKED_RECURRENCE_HISTORY
#if !FLA_OUTPUT_PACKED_DIRECT_READER
    publish_packed_recurrence_h_panel(
        stage,h,history_chunk,head,value_base,panel);
#endif
#if FLA_OUTPUT_AIU_QG
    if(threadIdx.x==0) wait_output_aiu();
#endif
#endif
#if FLA_OUTPUT_AIU_H && !FLA_OUTPUT_PACKED_RECURRENCE_HISTORY
#if FLA_OUTPUT_LEADER_AIU_WAIT
    if(threadIdx.x==0) wait_output_aiu();
#else
    wait_output_aiu();
#endif
#if !FLA_OUTPUT_NATIVE_H_LAYOUT
    __syncthreads();
    publish_h_panel_from_linear(stage);
#endif
#endif
#if FLA_OUTPUT_SPLIT_QG
#elif FLA_OUTPUT_VEC_PUBLISH && FLA_OUTPUT_AIU_H && !FLA_OUTPUT_NATIVE_H_LAYOUT
    publish_qg_panel_vec(stage,q,g,first,rows,head,panel,scale);
#elif !(FLA_OUTPUT_VEC_PUBLISH && FLA_OUTPUT_NATIVE_H_LAYOUT)
    publish_qg_h_panel(stage,q,g,h,first,rows,head,value_base,panel,
                       history_chunk,scale);
#endif
    __syncthreads();
#if FLA_OUTPUT_PACKED_RECURRENCE_HISTORY && FLA_OUTPUT_PACKED_DIRECT_READER
    mma_accumulate_packed_h(stage,mma,accumulator,h,history_chunk,head,value_base,panel);
#elif FLA_OUTPUT_NATIVE_H_LAYOUT
    mma_accumulate_native_h(stage,mma,accumulator);
#else
    mma_accumulate(stage,mma,accumulator);
#endif
  }

  // Every qg/H reader completes before both 8 KiB operands are republished.
  __syncthreads();
#if FLA_OUTPUT_VEC_PUBLISH
  publish_lower_A_and_v_vec(stage,A,v,first,rows,head,value_base);
#else
  publish_lower_A_and_v(stage,A,v,first,rows,head,value_base);
#endif
  __syncthreads();
  mma_accumulate(stage,mma,accumulator);
#if FLA_OUTPUT_BLOCKED_STORE
  store_blocked_output(
      stage,accumulator,output,first,rows,head,value_base);
#elif FLA_OUTPUT_PACKED_STORE
  store_packed_output(
      stage,accumulator,coordinates,output,first,rows,head,value_base);
#else
  store_output(accumulator,coordinates,output,first,rows,head,value_base);
#endif
#else
  __trap();
#endif
}

#if FLA_OUTPUT_BV128_PAIR_CTA
extern "C" __global__ __launch_bounds__(256,1) __block_size__((256,1,1))
void fla_glm53_output_h8_bv128_pair(
    uint16_t const* q,uint16_t const* v,float const* g,uint16_t const* h,
    uint16_t* output,uint16_t const* A,int32_t const* cu,
    int32_t const* indices,float scale,int total_tokens,int n) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  using namespace fla_output;
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& pair=*reinterpret_cast<PairStage*>(dynamic_shared);
  auto& pair_stage=*reinterpret_cast<Stage*>(dynamic_shared);
  int history_chunk=blockIdx.y;
  int head=blockIdx.z;
  int sequence=indices[2*history_chunk];
  int chunk=indices[2*history_chunk+1];
  if(sequence<0 || sequence>=n || chunk<0) return;
  int bos=cu[sequence],eos=cu[sequence+1];
  int64_t first=int64_t(cu[sequence])+int64_t(chunk)*kBt;
  if(bos<0 || eos>total_tokens || first<bos || first>=eos) return;
  int rows=min(kBt,eos-int(first));

  Mma mma;
  auto thr=mma.get_thread_slice(threadIdx.x);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},_64{})));
  auto accumulator0=Mma::make_fragment_C(coordinates);
  auto accumulator1=Mma::make_fragment_C(coordinates);
  clear(accumulator0);
  clear(accumulator1);

  CUTE_UNROLL
  for(int panel=0;panel<2;++panel) {
    if(panel!=0) __syncthreads();
    if(threadIdx.x==0)
      issue_pair_h_panel_aiu(pair.b0,h,history_chunk,head,0,panel);
#if FLA_OUTPUT_SPLIT_QG
    publish_precomputed_qg_panel_vec(
        pair_stage,q,first,rows,head,panel);
#else
    publish_qg_panel_vec(pair_stage,q,g,first,rows,head,panel,scale);
#endif
    wait_output_aiu();
    __syncthreads();

    if(threadIdx.x==0)
      issue_pair_h_panel_aiu(pair.b1,h,history_chunk,head,kBv,panel);
    mma_accumulate_pair<true>(pair.a,pair.b0,mma,accumulator0);
    wait_output_aiu();
    __syncthreads();
    mma_accumulate_pair<true>(pair.a,pair.b1,mma,accumulator1);
  }

  __syncthreads();
  publish_pair_A_and_v(pair,A,v,first,rows,head);
  __syncthreads();
  mma_accumulate_pair<false>(pair.a,pair.b0,mma,accumulator0);
  mma_accumulate_pair<false>(pair.a,pair.b1,mma,accumulator1);
  store_blocked_output(
      pair_stage,accumulator0,output,first,rows,head,0);
  store_blocked_output(
      pair_stage,accumulator1,output,first,rows,head,kBv);
#else
  __trap();
#endif
}
#endif

#if FLA_OUTPUT_BV128_PAIR_WARPGROUPS
extern "C" __global__ __launch_bounds__(512,1) __block_size__((512,1,1))
void fla_glm53_output_h8_bv128_warpgroups(
    uint16_t const* q,uint16_t const* v,float const* g,uint16_t const* h,
    uint16_t* output,uint16_t const* A,int32_t const* cu,
    int32_t const* indices,float scale,int total_tokens,int n) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  using namespace fla_output;
  extern __shared__ __align__(128) unsigned char dynamic_shared[];
  auto& pair=*reinterpret_cast<PairStage*>(dynamic_shared);
  auto& pair_stage=*reinterpret_cast<Stage*>(dynamic_shared);
  int group=threadIdx.x/kThreads;
  int local_tid=threadIdx.x%kThreads;
  int history_chunk=blockIdx.y;
  int head=blockIdx.z;
  int sequence=indices[2*history_chunk];
  int chunk=indices[2*history_chunk+1];
  if(sequence<0 || sequence>=n || chunk<0) return;
  int bos=cu[sequence],eos=cu[sequence+1];
  int64_t first=int64_t(cu[sequence])+int64_t(chunk)*kBt;
  if(bos<0 || eos>total_tokens || first<bos || first>=eos) return;
  int rows=min(kBt,eos-int(first));

  Mma mma;
  auto thr=mma.get_thread_slice(local_tid);
  auto coordinates=thr.partition_C(
      make_identity_tensor(make_shape(_64{},_64{})));
  auto accumulator=Mma::make_fragment_C(coordinates);
  clear(accumulator);

  CUTE_UNROLL
  for(int panel=0;panel<2;++panel) {
    if(panel!=0) __syncthreads();
    if(local_tid==0)
      issue_pair_h_panel_aiu(group==0 ? pair.b0 : pair.b1,
          h,history_chunk,head,group*kBv,panel);
    if(group==0) publish_qg_panel_vec(
        pair_stage,q,g,first,rows,head,panel,scale);
    wait_output_aiu();
    __syncthreads();
    mma_accumulate_warpgroup<true>(
        pair.a,group==0 ? pair.b0 : pair.b1,mma,accumulator,local_tid);
  }

  __syncthreads();
  publish_warpgroup_A_and_v(
      pair,A,v,first,rows,head,group,local_tid);
  __syncthreads();
  mma_accumulate_warpgroup<false>(
      pair.a,group==0 ? pair.b0 : pair.b1,mma,accumulator,local_tid);
  store_blocked_output_warpgroup(
      group==0 ? pair.b0 : pair.b1,accumulator,output,
      first,rows,head,group*kBv,local_tid);
#else
  __trap();
#endif
}
#endif

// Prepare-only opt-in. Never called from the graph-captured run/launch path.
extern "C" hggcError_t configure_fla_glm53_output_h8_bf16_d128_bt64() {
#if FLA_OUTPUT_BV128_PAIR_WARPGROUPS
  return hggcFuncSetAttribute(fla_glm53_output_h8_bv128_warpgroups,
      hggcFuncAttributeMaxDynamicSharedMemorySize,24576);
#elif FLA_OUTPUT_BV128_PAIR_CTA
  return hggcFuncSetAttribute(fla_glm53_output_h8_bv128_pair,
      hggcFuncAttributeMaxDynamicSharedMemorySize,24576);
#else
  return hggcFuncSetAttribute(fla_glm53_output_h8_bf16_d128_bt64,
      hggcFuncAttributeMaxDynamicSharedMemorySize,16384);
#endif
}

extern "C" hggcError_t launch_fla_glm53_output_h8_bf16_d128_bt64(
    void const* q,void const* v,float const* g,void* qg,void const* h,void* output,
    void const* A,int32_t const* cu,int32_t const* indices,float scale,
    int chunks,int total_tokens,int n,int heads,hggcStream_t stream) {
  if(chunks==0) return hggcSuccess;
#if FLA_OUTPUT_SPLIT_QG
#if !FLA_OUTPUT_EXTERNAL_QG
  fla_glm53_prepare_qg_h8_bf16_d128_bt64<<<dim3(chunks,1,heads),256,0,stream>>>(
      static_cast<uint16_t const*>(q),g,static_cast<uint16_t*>(qg),cu,indices,
      scale,total_tokens,n);
  hggcError_t qg_status=hggcGetLastError();
  if(qg_status!=hggcSuccess) return qg_status;
#endif
  q=qg;
#endif
#if FLA_OUTPUT_BV128_PAIR_WARPGROUPS
  fla_glm53_output_h8_bv128_warpgroups<<<dim3(1,chunks,heads),512,24576,stream>>>(
      static_cast<uint16_t const*>(q),static_cast<uint16_t const*>(v),g,
      static_cast<uint16_t const*>(h),static_cast<uint16_t*>(output),
      static_cast<uint16_t const*>(A),cu,indices,scale,total_tokens,n);
#elif FLA_OUTPUT_BV128_PAIR_CTA
  fla_glm53_output_h8_bv128_pair<<<dim3(1,chunks,heads),256,24576,stream>>>(
      static_cast<uint16_t const*>(q),static_cast<uint16_t const*>(v),g,
      static_cast<uint16_t const*>(h),static_cast<uint16_t*>(output),
      static_cast<uint16_t const*>(A),cu,indices,scale,total_tokens,n);
#else
  fla_glm53_output_h8_bf16_d128_bt64<<<dim3(2,chunks,heads),256,16384,stream>>>(
      static_cast<uint16_t const*>(q),static_cast<uint16_t const*>(v),g,
      static_cast<uint16_t const*>(h),static_cast<uint16_t*>(output),
      static_cast<uint16_t const*>(A),cu,indices,scale,total_tokens,n);
#endif
  return hggcGetLastError();
}
