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

// Standalone tc01 diagnostic: isolate the FlashQLA kkt Gram (K@K^T) bug.
//
// Reuses the REAL flashqla traits (flashqla_cute_traits.cuh -> flashqla_arch.cuh)
// and replicates KktKTraitsTc01 + kkt_load_k_tile_tc01 + the exact kkt_compute_head
// tc01 Gram GEMM, so any conclusion transfers directly to the kernel.
//
// It dumps two things for one known K[32,128] tile and compares against a CPU ref:
//   (a) K_buf read back through KktKTraitsTc01::SmemLayout view   -> load correctness
//   (b) the fp32 MMA accumulator of K@K^T                          -> read+MMA correctness
//
// Build (tc01 target = ppu_10):
//   hgcc -x cu -arch=ppu_10 ... tc01_gram_probe.cu -o tc01_gram_probe
#include <cstdio>
#include <cmath>
#include <hggc_runtime.h>

#include "flashqla_cute_traits.cuh"

using namespace cute;
namespace fq = flashqla;

constexpr int KKT_CS = 32;
constexpr int KKT_DK = 128;

// --- replica of KktKTraitsTc01 (kkt_solve_cute.cu) --------------------------
#if defined(FLASHQLA_ARCH_100)
template <typename Element>
struct KktKTraitsTc01 {
  using CuteElem = typename fq::ElementTraits<Element>::CuteType;
  static constexpr int BLK_MN = KKT_CS;
  static constexpr int BLK_K  = KKT_DK;
  static constexpr int BlockContSize   = BLK_K * static_cast<int>(sizeof(CuteElem));
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  static constexpr int AiuContElemSize = AiuContByteSize / static_cast<int>(sizeof(CuteElem));
  static constexpr int InstNum         = BLK_K / AiuContElemSize;
  static constexpr int BitsPerAIU      = BLK_MN * AiuContElemSize *
                                         static_cast<int>(sizeof(CuteElem)) * 8;
  using CopyInst = cute::PPU0010_AIU_LOAD<
      cute::C<BitsPerAIU>, CuteElem, /*Trans=*/false, /*Swzl=*/true>;
  using GmemTiledCopy = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<CopyInst, CuteElem>{},
      cute::Layout<cute::Shape <cute::_1, cute::_1>,
                   cute::Stride<cute::_1, cute::_1>>{},
      cute::Layout<cute::Shape<cute::Int<BLK_MN>, cute::Int<AiuContElemSize>>>{}));
  using SmemLayoutAtom = cute::Layout<
      cute::Shape<cute::_8, cute::Int<AiuContElemSize>>,
      cute::Stride<cute::Int<AiuContElemSize>, cute::_1>>;
  using SmemLayout = decltype(cute::tile_to_shape(
      SmemLayoutAtom{},
      cute::make_shape(cute::Int<BLK_MN>{}, cute::Int<BLK_K>{})));
  using SmemCopyOp = cute::PPU0010_TSM_LD_SWZL<
      CuteElem, BLK_MN, AiuContElemSize, /*Swap=*/false, /*Trans=*/false, InstNum>;
  using SmemCopyAtom = cute::Copy_Atom<SmemCopyOp, CuteElem>;
};

template <typename Element>
__forceinline__ __device__ void kkt_load_k_tile_tc01(
    Element* K_dst, const Element* K_src, int row_stride_elems, int rows, int tx) {
  using Traits = KktKTraitsTc01<Element>;
  using CuteElem = typename Traits::CuteElem;
  auto gA = cute::make_mix_tensor(
      cute::make_gmem_ptr(reinterpret_cast<const CuteElem*>(K_src)),
      cute::make_layout(
          cute::make_shape(cute::Int<Traits::BLK_MN>{}, cute::Int<Traits::BLK_K>{}),
          cute::make_counting_stride<2, 1, 1>()));
  auto sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElem*>(K_dst)),
      typename Traits::SmemLayout{});
  typename Traits::GmemTiledCopy tiled_copy;
  tiled_copy.desc_.template init<CuteElem, /*Trans=*/false,
                                  Traits::BLK_MN, Traits::BLK_K>(
      reinterpret_cast<const uint8_t*>(K_src),
      /*MN=*/rows, /*K=*/Traits::BLK_K,
      cute::make_stride(row_stride_elems, cute::_1{}));
  if (tx == 0) {
    auto thr = tiled_copy.get_slice(0);
    auto tAgA = thr.partition_S(gA);
    auto tAsA = thr.partition_D(sA);
    cute::copy(tiled_copy, tAgA, tAsA);
  }
}
#endif  // FLASHQLA_ARCH_100

// --- probe kernel -----------------------------------------------------------
template <typename Element>
__global__ void probe(const Element* Kin, float* out_kview, float* out_gram, int rows) {
#if defined(FLASHQLA_ARCH_100)
  using Traits = KktKTraitsTc01<Element>;
  using CuteElem = typename Traits::CuteElem;
  using cute::_;
  __shared__ Element K_buf[KKT_CS * KKT_DK];
  __shared__ float   gram_s[KKT_CS * KKT_CS];
  const int tx = threadIdx.x;

  kkt_load_k_tile_tc01<Element>(K_buf, Kin, KKT_DK, rows, tx);
  __syncthreads();
  if (tx == 0) asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  __syncthreads();
  asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
  __syncthreads();

  // (a) dump K_buf through the SmemLayout view
  auto sView = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElem*>(K_buf)),
      typename Traits::SmemLayout{});
  if (tx == 0) {
    for (int i = 0; i < KKT_CS; ++i)
      for (int j = 0; j < KKT_DK; ++j)
        out_kview[i * KKT_DK + j] = static_cast<float>(sView(i, j));
  }
  __syncthreads();

  // (b) Gram GEMM — EXACT copy of kkt_compute_head tc01 path
  auto tC = fq::make_acc<KKT_CS, KKT_CS, fq::TiledMmaM32T<Element>, Element>(tx);
  {
    auto sK = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<CuteElem*>(K_buf)),
        typename Traits::SmemLayout{});
    fq::TiledMmaM32T<Element> mma;
    auto thr_mma = mma.get_thread_slice(tx);
    auto rA_full = thr_mma.partition_fragment_A(sK);
    auto smem_copy_A = cute::make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
    auto thr_copy_A = smem_copy_A.get_thread_slice(tx);
    auto tAsA = thr_copy_A.partition_S(cute::make_mix_tensor_like(sK));
    auto tArA = thr_copy_A.retile_D(rA_full);
    auto rB_full = thr_mma.partition_fragment_B(sK);
    auto smem_copy_B = cute::make_tiled_copy_B(typename Traits::SmemCopyAtom{}, mma);
    auto thr_copy_B = smem_copy_B.get_thread_slice(tx);
    auto tBsB = thr_copy_B.partition_S(cute::make_mix_tensor_like(sK));
    auto tBrB = thr_copy_B.retile_D(rB_full);
    cute::clear(tC);
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      cute::copy(smem_copy_A, tAsA(_, _, k), tArA(_, _, k));
      cute::copy(smem_copy_B, tBsB(_, _, k), tBrB(_, _, k));
    }
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      cute::gemm(mma, rA_full(_, _, k), rB_full(_, _, k), tC);
    }
  }
  fq::store_acc_to_smem_fp32<fq::TiledMmaM32T<Element>>(
      tC, gram_s, KKT_CS, KKT_CS, KKT_CS, tx);
  __syncthreads();
  if (tx == 0) {
    for (int i = 0; i < KKT_CS * KKT_CS; ++i) out_gram[i] = gram_s[i];
  }
#else
  (void)Kin; (void)out_kview; (void)out_gram; (void)rows;
#endif  // FLASHQLA_ARCH_100
}

#if defined(FLASHQLA_ARCH_100)
// probe_trans: read K^T from the swizzled K_buf via Trans=true TSM, compute
// C = K^T @ I  -> C[dk,cs] must equal K[cs,dk].  M=DK=128, K(contraction)=CS=32.
template <typename Element>
__global__ void probe_trans(const Element* Kin, float* out_kt, int rows) {
  using Traits = KktKTraitsTc01<Element>;      // swzl write, CUBE_W=64, InstNum=2
  using CuteElem = typename Traits::CuteElem;
  using cute::_;
  constexpr int M = KKT_DK;   // 128
  constexpr int Kc = KKT_CS;  // 32 contraction
  __shared__ Element K_buf[KKT_CS * KKT_DK];
  __shared__ Element Ident[KKT_CS * KKT_CS];
  __shared__ float   C_s[KKT_DK * KKT_CS];
  const int tx = threadIdx.x;

  kkt_load_k_tile_tc01<Element>(K_buf, Kin, KKT_DK, rows, tx);
  __syncthreads();
  if (tx == 0) asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
  __syncthreads();
  asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
  // identity B[CS,CS] plain row-major
  for (int i = tx; i < KKT_CS * KKT_CS; i += blockDim.x) Ident[i] = __float2half((i/KKT_CS)==(i%KKT_CS)?1.f:0.f);
  __syncthreads();

  constexpr int CUBE_H = Traits::BLK_MN;         // 32
  constexpr int CUBE_W = Traits::AiuContElemSize; // 64
  using TsmTrans = cute::PPU0010_TSM_LD_SWZL<
      CuteElem, CUBE_H, CUBE_W, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/Traits::InstNum>;

  fq::TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tx);
  const int warp = tx / 32;
  const int warp_m = warp % 2;

  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElem*>(0x10)), fq::SmemLayout2D<M, Kc>{});
  auto rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    const int m_global = warp_m * 16 + m * 32;   // dk 0..127
    const int cube = m_global / CUBE_W;           // 64-wide cube
    const int dk_in_cube = m_global % CUBE_W;
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cs_col = k * 16;
      int afrag[4] = {0,0,0,0};
      TsmTrans::copy(afrag, reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(K_buf)),
                     /*coord_h=*/cs_col, /*coord_w=*/dk_in_cube,
                     /*cube_in_stage=*/cube, /*stage=*/0);
      auto rA = cute::recast<uint32_t>(rA_full(_, m, k));
      rA(0)=afrag[0]; rA(1)=afrag[1]; rA(2)=afrag[2]; rA(3)=afrag[3];
    }
  }
  auto sB = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<CuteElem*>(Ident)), fq::SmemLayout2D<Kc,Kc>{});
  auto rB_full = thr_mma.partition_fragment_B(sB);
  auto scpB = cute::make_tiled_copy_B(fq::SmemCopyAtomT<Element>{}, mma);
  auto tcpB = scpB.get_thread_slice(tx);
  auto tBsB = tcpB.partition_S(sB);
  auto tBrB = tcpB.retile_D(rB_full);
  auto tC = fq::make_acc<M, Kc, fq::TiledMmaM32T<Element>, Element>(tx);
  cute::clear(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) cute::copy(scpB, tBsB(_,_,k), tBrB(_,_,k));
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) cute::gemm(mma, rA_full(_,_,k), rB_full(_,_,k), tC);
  fq::store_acc_to_smem_fp32<fq::TiledMmaM32T<Element>>(tC, C_s, M, Kc, Kc, tx);
  __syncthreads();
  if (tx == 0) for (int i=0;i<M*Kc;i++) out_kt[i]=C_s[i];
}
#else
template <typename Element>
__global__ void probe_trans(const Element*, float*, int) {}
#endif

#define CK(x) do{ hggcError_t e=(x); if(e!=hggcSuccess){printf("HGGC err %s @ %d: %s\n",#x,__LINE__,hggcGetErrorString(e)); return 1;} }while(0)

int main(int argc, char** argv) {
  using Element = __half;
  const int rows = (argc > 1) ? atoi(argv[1]) : KKT_CS;   // active rows
  const int mode = (argc > 2) ? atoi(argv[2]) : 0;        // 0=structured 1=random-normalized

  float* hK = new float[KKT_CS * KKT_DK]();
  for (int i = 0; i < rows; ++i) {
    if (mode == 0) {
      for (int j = 0; j < KKT_DK; ++j) hK[i * KKT_DK + j] = (i * KKT_DK + j) * 0.001f;
    } else {
      float ss = 0.f;
      for (int j = 0; j < KKT_DK; ++j) { float v = sinf(0.3f*i + 0.11f*j) ; hK[i*KKT_DK+j]=v; ss+=v*v; }
      float inv = 1.f / sqrtf(ss);
      for (int j = 0; j < KKT_DK; ++j) hK[i*KKT_DK+j] *= inv;
    }
  }
  // fp16 round-trip on host to match device element type
  Element* hKe = new Element[KKT_CS * KKT_DK];
  for (int i = 0; i < KKT_CS * KKT_DK; ++i) hKe[i] = __float2half(hK[i]);
  float* hKf = new float[KKT_CS * KKT_DK];              // fp16-rounded ref
  for (int i = 0; i < KKT_CS * KKT_DK; ++i) hKf[i] = __half2float(hKe[i]);

  Element* dK; float *dKv, *dG;
  CK(hggcMalloc(&dK, sizeof(Element)*KKT_CS*KKT_DK));
  CK(hggcMalloc(&dKv, sizeof(float)*KKT_CS*KKT_DK));
  CK(hggcMalloc(&dG, sizeof(float)*KKT_CS*KKT_CS));
  CK(hggcMemcpy(dK, hKe, sizeof(Element)*KKT_CS*KKT_DK, hggcMemcpyHostToDevice));
  CK(hggcMemset(dKv, 0, sizeof(float)*KKT_CS*KKT_DK));
  CK(hggcMemset(dG, 0, sizeof(float)*KKT_CS*KKT_CS));

  probe<Element><<<1, 128>>>(dK, dKv, dG, rows);
  CK(hggcGetLastError());
  CK(hggcDeviceSynchronize());

  float* hKv = new float[KKT_CS*KKT_DK];
  float* hG  = new float[KKT_CS*KKT_CS];
  CK(hggcMemcpy(hKv, dKv, sizeof(float)*KKT_CS*KKT_DK, hggcMemcpyDeviceToHost));
  CK(hggcMemcpy(hG,  dG,  sizeof(float)*KKT_CS*KKT_CS, hggcMemcpyDeviceToHost));

  // (a) view vs input
  int kbad = 0; int first_i=-1, first_j=-1;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < KKT_DK; ++j) {
      float e = hKf[i*KKT_DK+j], g = hKv[i*KKT_DK+j];
      if (fabsf(e-g) > 1e-3f) { if(kbad==0){first_i=i;first_j=j;} kbad++; }
    }
  printf("[rows=%d mode=%d] VIEW: %d/%d mismatches", rows, mode, kbad, rows*KKT_DK);
  if (kbad) printf("  first bad (%d,%d) exp=%.4f got=%.4f", first_i, first_j,
                   hKf[first_i*KKT_DK+first_j], hKv[first_i*KKT_DK+first_j]);
  printf("\n  view row0[0..7]:"); for(int j=0;j<8;j++) printf(" %.3f", hKv[j]);
  printf("\n  ref  row0[0..7]:"); for(int j=0;j<8;j++) printf(" %.3f", hKf[j]);
  printf("\n  view row1[0..7]:"); for(int j=0;j<8;j++) printf(" %.3f", hKv[KKT_DK+j]);
  printf("\n  ref  row1[0..7]:"); for(int j=0;j<8;j++) printf(" %.3f", hKf[KKT_DK+j]);
  printf("\n");

  // (b) Gram vs CPU ref (fp16 inputs, fp32 accumulate)
  int gbad = 0; int gi=-1, gj=-1; double maxrel=0;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < rows; ++j) {
      double acc = 0;
      for (int k = 0; k < KKT_DK; ++k) acc += (double)hKf[i*KKT_DK+k]*(double)hKf[j*KKT_DK+k];
      float ref = (float)acc, got = hG[i*KKT_CS+j];
      double rel = fabs(ref-got)/(fabs(ref)+1e-6);
      if (rel > 2e-2) { if(gbad==0){gi=i;gj=j;} gbad++; }
      if (rel > maxrel) maxrel = rel;
    }
  printf("[rows=%d mode=%d] GRAM: %d/%d mismatches maxrel=%.3e", rows, mode, gbad, rows*rows, maxrel);
  if (gbad) printf("  first bad (%d,%d)", gi, gj);
  printf("\n");
  // Diagnostic corner entries
  auto cpu_g = [&](int i,int j){ double a=0; for(int k=0;k<KKT_DK;k++) a+=(double)hKf[i*KKT_DK+k]*(double)hKf[j*KKT_DK+k]; return (float)a; };
  printf("  G(0,0) got=%.4f ref=%.4f | G(1,0) got=%.4f ref=%.4f | G(1,1) got=%.4f ref=%.4f | G(2,0) got=%.4f ref=%.4f\n",
         hG[0], cpu_g(0,0), hG[1*KKT_CS+0], cpu_g(1,0), hG[1*KKT_CS+1], cpu_g(1,1), hG[2*KKT_CS+0], cpu_g(2,0));
  printf("RESULT view_ok=%d gram_ok=%d\n", kbad==0, gbad==0);

  // --- Trans=true K^T read validation: C[dk,cs] should equal K[cs,dk] ---
  float* dKT; CK(hggcMalloc(&dKT, sizeof(float)*KKT_DK*KKT_CS));
  CK(hggcMemset(dKT, 0, sizeof(float)*KKT_DK*KKT_CS));
  probe_trans<Element><<<1,128>>>(dK, dKT, rows);
  CK(hggcGetLastError()); CK(hggcDeviceSynchronize());
  float* hKT = new float[KKT_DK*KKT_CS];
  CK(hggcMemcpy(hKT, dKT, sizeof(float)*KKT_DK*KKT_CS, hggcMemcpyDeviceToHost));
  int tbad=0, ti=-1, tj=-1; double tmax=0;
  for (int dk=0; dk<KKT_DK; ++dk)
    for (int cs=0; cs<rows; ++cs) {
      float ref = hKf[cs*KKT_DK+dk], got = hKT[dk*KKT_CS+cs];
      double rel = fabs(ref-got)/(fabs(ref)+1e-3);
      if (rel > 2e-2) { if(tbad==0){ti=dk;tj=cs;} tbad++; }
      if (rel>tmax) tmax=rel;
    }
  printf("[rows=%d mode=%d] KT(trans): %d/%d mismatches maxrel=%.3e", rows, mode, tbad, KKT_DK*rows, tmax);
  if (tbad) printf("  first bad KT(dk=%d,cs=%d) got=%.4f exp=%.4f", ti,tj, hKT[ti*KKT_CS+tj], hKf[tj*KKT_DK+ti]);
  printf("\n  KT(0,0)=%.4f expK(0,0)=%.4f | KT(1,0)=%.4f expK(0,1)=%.4f | KT(64,1)=%.4f expK(1,64)=%.4f\n",
         hKT[0*KKT_CS+0], hKf[0*KKT_DK+0], hKT[1*KKT_CS+0], hKf[0*KKT_DK+1], hKT[64*KKT_CS+1], hKf[1*KKT_DK+64]);
  printf("RESULT_TRANS kt_ok=%d\n", tbad==0);
  return 0;
}
