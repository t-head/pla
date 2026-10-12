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
}  // namespace

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
