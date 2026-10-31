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

#include <hggc_runtime.h>
#include <cstdint>

namespace {
// A single fixed CTA performs a segmented scan. No D2H, dynamic allocation or cross-call cache; replay reads the current offsets.
template<class Offset>
__global__ void scan_chunks(Offset const* input, int32_t* cu, int64_t* offsets,
                            int n, int tokens, int capacity) {
  __shared__ int64_t scan[256];
  __shared__ int64_t carry;
  int tid=threadIdx.x;
  if(tid==0) {
    carry=0;
    offsets[0]=0;
    if(input[0]!=0) __trap();
  }
  __syncthreads();
  for(int base=0;base<n;base+=256) {
    int seq=base+tid;
    int64_t count=0;
    if(seq<n) {
      int64_t bos=input[seq],eos=input[seq+1];
      if(bos<0 || eos<bos || eos>tokens) __trap();
      cu[seq]=static_cast<int32_t>(bos);
      if(seq==n-1) cu[n]=static_cast<int32_t>(eos);
      count=(eos-bos+63)/64;
    }
    scan[tid]=count;
    __syncthreads();
    for(int step=1;step<256;step*=2) {
      int64_t prior=tid>=step ? scan[tid-step] : 0;
      __syncthreads();
      scan[tid]+=prior;
      __syncthreads();
    }
    if(seq<n) offsets[seq+1]=carry+scan[tid];
    __syncthreads();
    if(tid==0) {
      carry+=scan[255];
      if(carry>capacity) __trap();
    }
    __syncthreads();
  }
}

__global__ void fill_chunks(int64_t const* offsets, int32_t* indices,
                            int32_t* slots, int n, int capacity) {
  int index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index<n) slots[index]=index;
  if(index>=capacity) return;
  int64_t total=offsets[n];
  if(index>=total) {
    indices[2*index]=-1;
    indices[2*index+1]=-1;
    return;
  }
  int left=0,right=n;
  while(left<right) {
    int mid=left+(right-left)/2;
    if(index<offsets[mid+1]) right=mid;
    else left=mid+1;
  }
  indices[2*index]=left;
  indices[2*index+1]=static_cast<int32_t>(index-offsets[left]);
}

// N1 device-offset CP descriptor. The launch keeps a fixed segment count for
// Graph stability; replay derives the geometry from the current effective
// length. Lengths outside [min_tokens,max_tokens] place every chunk in
// segment zero, so trailing empty segments only forward the exact BF16 state.
__global__ void cp_device_geometry(int32_t const* cu, int32_t* bounds,
                                   int64_t* offsets, int split, int width,
                                   int min_tokens, int max_tokens) {
  if(threadIdx.x!=0) return;
  int64_t tokens=int64_t(cu[1])-cu[0];
  int64_t chunks=(tokens+63)/64;
  bool spread=tokens>=min_tokens && tokens<=max_tokens && chunks>=split;
  int64_t quotient=spread ? chunks/split : 0;
  int64_t remainder=spread ? chunks%split : 0;
  int64_t chunk_offset=0;
  bounds[0]=cu[0];
  offsets[0]=0;
  for(int segment=0;segment<width-1;++segment) {
    if(segment<split)
      chunk_offset=spread ? chunk_offset+quotient+(segment<remainder ? 1 : 0)
                          : chunks;
    offsets[segment+1]=chunk_offset;
    int64_t end=64*chunk_offset<tokens ? 64*chunk_offset : tokens;
    bounds[segment+1]=static_cast<int32_t>(cu[0]+end);
  }
}

// Device-offset output tail: rows [cu[n], tokens) are not produced by any
// stage and must read +0. scan_chunks traps unless cu[0] == 0, so the tail is
// the only uncovered range; a full-length replay stores nothing.
__global__ void zero_output_tail(int32_t const* cu, int n, uint4* out,
                                 int64_t row_vectors, int64_t tokens) {
  int64_t first=int64_t(cu[n])*row_vectors;
  int64_t last=tokens*row_vectors;
  int64_t stride=int64_t(gridDim.x)*blockDim.x;
  for(int64_t i=first+int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<last;i+=stride)
    out[i]=make_uint4(0,0,0,0);
}

// N1 has no segmented scan dependency: every CTA reads the current input
// pair and derives its own bounds. Only CTA0 publishes metadata; other CTAs
// never read that publication, so no cross-CTA synchronization is needed.
template<class Offset>
__global__ void metadata_n1_tail(Offset const* input, int32_t* cu,
    int64_t* offsets, int32_t* indices, int32_t* slots, int tokens,
    int capacity, int32_t* cp_bounds, int64_t* cp_offsets, int split,
    int width, int min_tokens, int max_tokens, uint4* out,
    int64_t row_vectors) {
  const int64_t bos=input[0],eos=input[1];
  if(bos!=0 || eos<bos || eos>tokens) __trap();
  const int64_t chunks=(eos-bos+63)/64;
  if(chunks>capacity) __trap();
  if(blockIdx.x==0 && threadIdx.x==0) {
    cu[0]=static_cast<int32_t>(bos);
    cu[1]=static_cast<int32_t>(eos);
    offsets[0]=0; offsets[1]=chunks; slots[0]=0;
    if(split>1) {
      const bool spread=eos-bos>=min_tokens && eos-bos<=max_tokens && chunks>=split;
      const int64_t quotient=spread ? chunks/split : 0;
      const int64_t remainder=spread ? chunks%split : 0;
      int64_t chunk_offset=0;
      cp_bounds[0]=static_cast<int32_t>(bos); cp_offsets[0]=0;
      for(int segment=0;segment<width-1;++segment) {
        if(segment<split)
          chunk_offset=spread ? chunk_offset+quotient+(segment<remainder ? 1 : 0) : chunks;
        cp_offsets[segment+1]=chunk_offset;
        const int64_t end=64*chunk_offset<eos-bos ? 64*chunk_offset : eos-bos;
        cp_bounds[segment+1]=static_cast<int32_t>(bos+end);
      }
    }
  }
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  const int stride=gridDim.x*blockDim.x;
  for(int i=index;i<capacity;i+=stride) {
    indices[2*i]=i<chunks ? 0 : -1;
    indices[2*i+1]=i<chunks ? i : -1;
  }
  const int64_t first=eos*row_vectors,last=int64_t(tokens)*row_vectors;
  for(int64_t i=first+index;i<last;i+=stride) out[i]=make_uint4(0,0,0,0);
}
}  // namespace

hggcError_t launch_fla_zero_output_tail(int32_t const* cu, int n, void* out,
    int64_t row_bytes, int64_t tokens, hggcStream_t stream) {
  if(!cu || !out || n<1 || tokens<1 || row_bytes<16 || row_bytes%16 ||
     reinterpret_cast<uintptr_t>(out)%16)
    return hggcErrorInvalidValue;
  const int64_t vectors=row_bytes/16*tokens;
  const int64_t blocks=(vectors+255)/256;
  zero_output_tail<<<static_cast<unsigned>(blocks<512 ? blocks : 512),256,0,stream>>>(
      cu,n,static_cast<uint4*>(out),row_bytes/16,tokens);
  return hggcGetLastError();
}

hggcError_t launch_fla_metadata(void const* input, bool wide, int32_t* cu,
    int64_t* offsets, int32_t* indices, int32_t* slots, int n, int tokens,
    int capacity, hggcStream_t stream) {
  if(wide) scan_chunks<<<1,256,0,stream>>>(static_cast<int64_t const*>(input),
      cu,offsets,n,tokens,capacity);
  else scan_chunks<<<1,256,0,stream>>>(static_cast<int32_t const*>(input),
      cu,offsets,n,tokens,capacity);
  auto status=hggcGetLastError();
  if(status!=hggcSuccess) return status;
  int count=n>capacity ? n : capacity;
  fill_chunks<<<(count+255)/256,256,0,stream>>>(offsets,indices,slots,n,capacity);
  return hggcGetLastError();
}

hggcError_t launch_fla_cp_device_geometry(int32_t const* cu, int32_t* bounds,
    int64_t* offsets, int split, int width, int min_tokens, int max_tokens,
    hggcStream_t stream) {
  if(!cu || !bounds || !offsets || split<1 || width<split+1 || min_tokens<1 ||
     max_tokens<min_tokens)
    return hggcErrorInvalidValue;
  cp_device_geometry<<<1,32,0,stream>>>(cu,bounds,offsets,split,width,
                                        min_tokens,max_tokens);
  return hggcGetLastError();
}

hggcError_t launch_fla_metadata_n1_tail(void const* input, bool wide,
    int32_t* cu, int64_t* offsets, int32_t* indices, int32_t* slots,
    int tokens, int capacity, int32_t* cp_bounds, int64_t* cp_offsets,
    int split, int width, int min_tokens, int max_tokens, void* out,
    int64_t row_bytes, hggcStream_t stream) {
  if(!input || !cu || !offsets || !indices || !slots || !out || tokens<1 ||
      capacity<1 || row_bytes<16 || row_bytes%16 ||
      reinterpret_cast<uintptr_t>(out)%16 || split<0 ||
      (split>1 && (!cp_bounds || !cp_offsets || width<split+1 ||
                    min_tokens<1 || max_tokens<min_tokens)))
    return hggcErrorInvalidValue;
  const int64_t vectors=row_bytes/16*tokens;
  const int64_t extent=vectors>capacity ? vectors : capacity;
  const int64_t blocks=(extent+255)/256;
  const unsigned grid=static_cast<unsigned>(blocks<512 ? blocks : 512);
  if(wide) metadata_n1_tail<<<grid,256,0,stream>>>(
      static_cast<int64_t const*>(input),cu,offsets,indices,slots,tokens,
      capacity,cp_bounds,cp_offsets,split,width,min_tokens,max_tokens,
      static_cast<uint4*>(out),row_bytes/16);
  else metadata_n1_tail<<<grid,256,0,stream>>>(
      static_cast<int32_t const*>(input),cu,offsets,indices,slots,tokens,
      capacity,cp_bounds,cp_offsets,split,width,min_tokens,max_tokens,
      static_cast<uint4*>(out),row_bytes/16);
  return hggcGetLastError();
}
