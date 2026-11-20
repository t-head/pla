// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include "precision_repair_api.h"
#include <hggc_bf16.h>
#include <hggc_fp16.h>

namespace flashqla_precision {
namespace {
template <typename E> __device__ __forceinline__ float load(E x);
template <> __device__ __forceinline__ float load(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float load(__ppu_bfloat16 x) { return __bfloat162float(x); }
template <typename E> __device__ __forceinline__ E convert(float x);
template <> __device__ __forceinline__ __half convert(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __ppu_bfloat16 convert(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ double warp_sum(double x) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}
__device__ __forceinline__ double multiply_add(double a, double b, double c) { return fma(a, b, c); }
__device__ __forceinline__ double exponential(double x) { return exp(x); }

// A structural fallback class, not a general floating-point error bound.
// Products of finite FP16/BF16 values are exact and in range in FP64, so this
// cross-product test has no tolerance or rounded-Gram dependency. Disjoint
// support and roundoff-uncertain orthogonality are additional content classes.
template <typename E>
__device__ __forceinline__ bool structural_pair(
    const E* k, int first, int second, int key_heads, int head, int lane) {
    float af[4], bf[4], dot32 = 0.f, absolute32 = 0.f;
    bool exact_products = true;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dk = lane+32*j;
        af[j] = load(k[(int64_t(first)*key_heads+head)*128+dk]);
        bf[j] = load(k[(int64_t(second)*key_heads+head)*128+dk]);
        const float product = __fmul_rn(af[j],bf[j]);
        const uint32_t am = __float_as_uint(af[j])&0x7fffffffu;
        const uint32_t bm = __float_as_uint(bf[j])&0x7fffffffu;
        const uint32_t pm = __float_as_uint(product)&0x7fffffffu;
        exact_products = exact_products && am < 0x7f800000u && bm < 0x7f800000u &&
            ((am == 0 || bm == 0) || (pm >= 0x00800000u && pm < 0x7f800000u));
        dot32 = __fadd_rn(dot32,product);
        absolute32 = __fadd_rn(absolute32,__uint_as_float(pm));
    }
    const float a1 = __shfl_sync(0xffffffff,af[0],1);
    const float b1 = __shfl_sync(0xffffffff,bf[0],1);
    bool nonparallel = false;
    if (lane == 0) {
        const float a0 = af[0], b0 = bf[0];
        const uint32_t a0m = __float_as_uint(a0)&0x7fffffffu;
        const uint32_t a1m = __float_as_uint(a1)&0x7fffffffu;
        const uint32_t b0m = __float_as_uint(b0)&0x7fffffffu;
        const uint32_t b1m = __float_as_uint(b1)&0x7fffffffu;
        const bool finite_inputs =
            (a0m == 0 || (a0m >= 0x00800000u && a0m < 0x7f800000u)) &&
            (a1m == 0 || (a1m >= 0x00800000u && a1m < 0x7f800000u)) &&
            (b0m == 0 || (b0m >= 0x00800000u && b0m < 0x7f800000u)) &&
            (b1m == 0 || (b1m >= 0x00800000u && b1m < 0x7f800000u));
        const float p = __fmul_rn(a0,b1), q = __fmul_rn(a1,b0);
        const uint32_t pm = __float_as_uint(p)&0x7fffffffu;
        const uint32_t qm = __float_as_uint(q)&0x7fffffffu;
        const bool exact_p = a0m == 0 || b1m == 0 || (pm >= 0x00800000u && pm < 0x7f800000u);
        const bool exact_q = a1m == 0 || b0m == 0 || (qm >= 0x00800000u && qm < 0x7f800000u);
        const bool overlap = (a0m != 0 && b0m != 0) || (a1m != 0 && b1m != 0);
        nonparallel = finite_inputs && exact_p && exact_q && overlap && p != q;
    }
    nonparallel = __any_sync(0xffffffff,nonparallel);
    exact_products = __all_sync(0xffffffff,exact_products);
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        dot32 = __fadd_rn(dot32,__shfl_xor_sync(0xffffffff,dot32,offset));
        absolute32 = __fadd_rn(absolute32,__shfl_xor_sync(0xffffffff,absolute32,offset));
    }
    // Normal products are exact (<=22 significant bits). Four local terms
    // plus five warp additions have <=9 roundings/path. 16*u32 covers both
    // reductions' relative error; 128*FLT_MIN covers all possible FTZ loss.
    const float bound32 = __fadd_rn(__fmul_rn(absolute32,0x1p-20f),0x1p-119f);
    const bool finite_sums = (__float_as_uint(dot32)&0x7f800000u) != 0x7f800000u &&
                             (__float_as_uint(absolute32)&0x7f800000u) != 0x7f800000u;
    // This certified rejection is disabled in the separate full-FP64 parity build.
    const bool reject = nonparallel && exact_products && finite_sums && fabsf(dot32) > bound32;
    if (__all_sync(0xffffffff,reject)) return false;

    double a[4], b[4];
    bool finite = true, nonzero_a = false, nonzero_b = false, overlap = false;
    int pivot = 128;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dk = lane + 32*j;
        const float av = af[j], bv = bf[j];
        a[j] = double(av); b[j] = double(bv);
        finite = finite && (__float_as_uint(av) & 0x7f800000u) != 0x7f800000u
                        && (__float_as_uint(bv) & 0x7f800000u) != 0x7f800000u;
        nonzero_a = nonzero_a || a[j] != 0.;
        nonzero_b = nonzero_b || b[j] != 0.;
        overlap = overlap || (a[j] != 0. && b[j] != 0.);
        if (a[j] != 0. && dk < pivot) pivot = dk;
    }
    if (__any_sync(0xffffffff, !finite) ||
        !__any_sync(0xffffffff, nonzero_a) ||
        !__any_sync(0xffffffff, nonzero_b)) return true;
    if (!__any_sync(0xffffffff,overlap)) return true;
    double dot64 = 0., absolute64 = 0.;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const double product = a[j]*b[j];
        dot64 += product;
        absolute64 += fabs(product);
    }
    dot64 = warp_sum(dot64);
    absolute64 = warp_sum(absolute64);
    // 16*u64 safely encloses the same reduction's rounding error. The test
    // selects every exact orthogonal pair, including cancellation residuals;
    // this is a roundoff interval, not an empirical similarity threshold.
    if (__any_sync(0xffffffff,fabs(dot64) <= absolute64*0x1p-49)) return true;
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        const int other = __shfl_xor_sync(0xffffffff, pivot, offset);
        pivot = pivot < other ? pivot : other;
    }
    double pa = 0., pb = 0.;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (pivot / 32 == j) { pa = a[j]; pb = b[j]; }
    }
    pa = __shfl_sync(0xffffffff, pa, pivot & 31);
    pb = __shfl_sync(0xffffffff, pb, pivot & 31);
    bool parallel = true;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        parallel = parallel && a[j] * pb == b[j] * pa;
    return __all_sync(0xffffffff, parallel);
}

// Isolated scheduling/data-movement variants; classifier and FP64 math unchanged.
__global__ void guard_init_parallel_kernel(const int32_t* cu,int32_t* flags,
    int tokens,int key_heads,int sequences,bool include_empty) {
    const int64_t tasks=int64_t(key_heads)*sequences;
    for(int64_t task=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        task<tasks;task+=int64_t(gridDim.x)*blockDim.x){
        const int seq=int(task/key_heads),bos=cu[seq],eos=cu[seq+1];
        if(cu[0]!=0||bos<0||eos<bos||eos>tokens)__trap();
        const int length=eos-bos;
        flags[task]=length<=3&&(length>0||include_empty);
    }
}

template <typename E>
__global__ __launch_bounds__(128) void guard_windows_parallel_kernel(
    const E* k,const int32_t* cu,int32_t* flags,int tokens,int key_heads,
    int sequences,const float* beta,int heads) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,head=blockIdx.y;
    const int64_t window=(int64_t(blockIdx.x)*4+warp)*32;
    const int last=cu[sequences];
    if(window>=tokens||window>=last)return;
    const int64_t end=window+32<last?window+32:last;
    int seq=0;
    if(lane==0){
        int lo=0,hi=sequences;
        while(lo<hi){
            const int mid=int((int64_t(lo)+hi+1)/2);
            if(cu[mid]<=window)lo=mid;else hi=mid-1;
        }
        seq=lo;
    }
    seq=__shfl_sync(0xffffffff,seq,0);
    for(;seq<sequences&&cu[seq]<end;++seq){
        const int bos=cu[seq],eos=cu[seq+1];
        if(eos-bos<=3)continue;
        const int64_t first=int64_t(bos)+((window>bos?window-bos:0)+31)/32*32;
        if(first>=eos||first>=end)continue;
        bool saturated=false;
        if(beta&&first+lane<eos){
            for(int vh=head*(heads/key_heads);vh<(head+1)*(heads/key_heads);++vh)
                saturated|=beta[(first+lane)*heads+vh]==1.f;
        }
        saturated=__any_sync(0xffffffff,saturated);
        const bool selected=saturated||(first+1<eos&&
            structural_pair(k,int(first),int(first+1),key_heads,head,lane));
        if(selected&&lane==0)atomicOr(flags+int64_t(seq)*key_heads+head,1);
    }
}


template <typename E>
__global__ __launch_bounds__(128) void guard_kernel(
    const E* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, bool include_empty, const float* beta, int heads) {
    const int task = blockIdx.x, seq = task / key_heads, head = task % key_heads;
    const int bos = cu[seq], eos = cu[seq+1];
    if (cu[0] != 0 || bos < 0 || eos < bos || eos > tokens) __trap();
    const int length = int(int64_t(eos)-bos);
    if (length <= 3) {
        if (threadIdx.x == 0) flags[task] = length > 0 || include_empty;
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int selected = 0;
    for (int chunk = warp; chunk < (int64_t(length)+31)/32; chunk += 4) {
        const int first = bos + chunk*32;
        bool saturated = false;
        if (beta && int64_t(first)+lane < eos) {
            for (int value_head = head*(heads/key_heads);
                 value_head < (head+1)*(heads/key_heads); ++value_head)
                saturated |= beta[(int64_t(first)+lane)*heads+value_head] == 1.f;
        }
        saturated = __any_sync(0xffffffff,saturated);
        if (saturated || (first+1 < eos &&
            structural_pair(k, first, first+1, key_heads, head, lane))) {
            selected = 1;
            break;
        }
    }
    __shared__ int warp_selected[4];
    if (lane == 0) warp_selected[warp] = selected;
    __syncthreads();
    if (threadIdx.x == 0)
        flags[task] = warp_selected[0] | warp_selected[1] | warp_selected[2] | warp_selected[3];
}

__global__ __launch_bounds__(256) void snapshot_tiled_parallel_kernel(
    const void* initial,float* snapshot,const int32_t* indices,const int32_t* cu,
    int sequences,int heads,int slots,bool bf16,const int32_t* flags,int key_heads) {
    __shared__ float transpose[32][33];
    const int64_t tasks=int64_t(sequences)*heads*16;
    for(int64_t task=blockIdx.x;task<tasks;task+=gridDim.x){
        const int64_t sh=task/16;
        const int seq=int(sh/heads),head=int(sh%heads),tile=int(task%16);
        if(flags&&!flags[seq*key_heads+head/(heads/key_heads)])continue;
        const int dv0=(tile/4)*32,dk0=(tile%4)*32;
        const bool empty=indices&&cu[seq]==cu[seq+1];
        const int slot=empty?0:indices?indices[seq]:seq;
        if(!empty&&(slot<0||slot>=slots))__trap();
        const int64_t base=(int64_t(slot)*heads+head)*128*128;
        if(indices){
            for(int i=threadIdx.x;i<1024;i+=256){
                const int dv=dv0+i/32,dk=dk0+i%32;
                const int64_t src=base+dv*128+dk;
                snapshot[sh*128*128+dv*128+dk]=empty?0.f:
                    bf16?load(static_cast<const __ppu_bfloat16*>(initial)[src]):
                         static_cast<const float*>(initial)[src];
            }
        }else{
            for(int i=threadIdx.x;i<1024;i+=256){
                const int64_t src=base+(dk0+i/32)*128+dv0+i%32;
                transpose[i/32][i%32]=bf16?
                    load(static_cast<const __ppu_bfloat16*>(initial)[src]):
                    static_cast<const float*>(initial)[src];
            }
            __syncthreads();
            for(int i=threadIdx.x;i<1024;i+=256)
                snapshot[sh*128*128+(dv0+i/32)*128+dk0+i%32]=transpose[i%32][i/32];
            __syncthreads();
        }
    }
}

template <typename E, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void guard_snapshot_medium_kernel(
    const E* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, bool include_empty, const float* beta, int heads,
    const void* initial, float* snapshot, const int32_t* indices,
    int slots, bool state_bf16) {
    const int task = blockIdx.x % key_heads, seq = blockIdx.y, head = task;
    const int shard = blockIdx.x / key_heads, shards = gridDim.x / key_heads;
    const int bos = cu[seq], eos = cu[seq+1];
    if (cu[0] != 0 || bos < 0 || eos < bos || eos > tokens) __trap();
    const int length = int(int64_t(eos)-bos);
    int copy_selected = length > 0 || include_empty;
    if (length > 3) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int selected = 0;
    for (int chunk = warp; chunk < (int64_t(length)+31)/32; chunk += WARPS) {
        const int first = bos + chunk*32;
        bool saturated = false;
        if (beta && int64_t(first)+lane < eos) {
            for (int value_head = head*(heads/key_heads);
                 value_head < (head+1)*(heads/key_heads); ++value_head)
                saturated |= beta[(int64_t(first)+lane)*heads+value_head] == 1.f;
        }
        saturated = __any_sync(0xffffffff,saturated);
        if (saturated || (first+1 < eos &&
            structural_pair(k, first, first+1, key_heads, head, lane))) {
            selected = 1;
            break;
        }
    }
    __shared__ int warp_selected[WARPS];
    if (lane == 0) warp_selected[warp] = selected;
    __syncthreads();
        copy_selected = 0;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) copy_selected |= warp_selected[w];
    }
    if (shard == 0 && threadIdx.x == 0) flags[int64_t(seq)*key_heads + head] = copy_selected;
    if (!copy_selected) return;
    // Every thread owns disjoint DV,DK elements of every value head sharing
    // this key head. Stream order completes the snapshot before fast writers.
    const bool empty = indices && bos == eos;
    const int slot = empty ? 0 : (indices ? indices[seq] : seq);
    if (!empty && (slot < 0 || slot >= slots)) __trap();
    const int heads_per_k = heads / key_heads;
    for (int64_t i = int64_t(shard)*blockDim.x+threadIdx.x; i < int64_t(heads_per_k)*128*128; i += int64_t(shards)*blockDim.x) {
        const int value_head = head*heads_per_k + int(i/(128*128));
        const int dv = int((i/128)%128), dk = int(i%128);
        const int64_t out = (int64_t(seq)*heads+value_head)*128*128+dv*128+dk;
        const int64_t src = (int64_t(slot)*heads+value_head)*128*128 +
                            (indices ? dv*128+dk : dk*128+dv);
        snapshot[out] = empty ? 0.f : state_bf16
            ? load(static_cast<const __ppu_bfloat16*>(initial)[src])
            : static_cast<const float*>(initial)[src];
    }
}

template <typename E>
__global__ __launch_bounds__(128) void guard_snapshot_short_kernel(
    const E* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, bool include_empty, const float* beta, int heads,
    const void* initial, float* snapshot, const int32_t* indices,
    int slots, bool state_bf16) {
    const int task = blockIdx.x % key_heads, seq = 0, head = task;
    const int shard = blockIdx.x / key_heads, shards = gridDim.x / key_heads;
    const int bos = cu[seq], eos = cu[seq+1];
    if (cu[0] != 0 || bos < 0 || eos < bos || eos > tokens) __trap();
    const int length = int(int64_t(eos)-bos);
    int copy_selected = length > 0 || include_empty;
    if (length > 3) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int selected = 0;
    for (int chunk = warp; chunk < (int64_t(length)+31)/32; chunk += 4) {
        const int first = bos + chunk*32;
        bool saturated = false;
        if (beta && int64_t(first)+lane < eos) {
            for (int value_head = head*(heads/key_heads);
                 value_head < (head+1)*(heads/key_heads); ++value_head)
                saturated |= beta[(int64_t(first)+lane)*heads+value_head] == 1.f;
        }
        saturated = __any_sync(0xffffffff,saturated);
        if (saturated || (first+1 < eos &&
            structural_pair(k, first, first+1, key_heads, head, lane))) {
            selected = 1;
            break;
        }
    }
    __shared__ int warp_selected[4];
    if (lane == 0) warp_selected[warp] = selected;
    __syncthreads();
        copy_selected = warp_selected[0] | warp_selected[1] |
                        warp_selected[2] | warp_selected[3];
    }
    if (shard == 0 && threadIdx.x == 0) flags[task] = copy_selected;
    if (!copy_selected) return;
    // Every thread owns disjoint DV,DK elements of every value head sharing
    // this key head. Stream order completes the snapshot before fast writers.
    const bool empty = indices && bos == eos;
    const int slot = empty ? 0 : (indices ? indices[seq] : seq);
    if (!empty && (slot < 0 || slot >= slots)) __trap();
    const int heads_per_k = heads / key_heads;
    for (int64_t i = int64_t(shard)*128+threadIdx.x; i < int64_t(heads_per_k)*128*128; i += int64_t(shards)*128) {
        const int value_head = head*heads_per_k + int(i/(128*128));
        const int dv = int((i/128)%128), dk = int(i%128);
        const int64_t out = (int64_t(seq)*heads+value_head)*128*128+dv*128+dk;
        const int64_t src = (int64_t(slot)*heads+value_head)*128*128 +
                            (indices ? dv*128+dk : dk*128+dv);
        snapshot[out] = empty ? 0.f : state_bf16
            ? load(static_cast<const __ppu_bfloat16*>(initial)[src])
            : static_cast<const float*>(initial)[src];
    }
}

// Dense call-local FP32 [sequence,head,DV,DK]. Metadata is read on each replay.
__global__ void snapshot_kernel(const void* initial, float* snapshot,
    const int32_t* indices, const int32_t* cu, int sequences, int heads, int slots, bool bf16,
    const int32_t* flags, int key_heads) {
    const int64_t size = int64_t(sequences) * heads * 128 * 128;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += int64_t(gridDim.x) * blockDim.x) {
        const int dk = i % 128, dv = (i / 128) % 128;
        const int head = (i / (128 * 128)) % heads;
        const int seq = i / (int64_t(heads) * 128 * 128);
        if (flags && !flags[seq*key_heads+head/(heads/key_heads)]) continue;
        // The indexed fast path skips empty requests, including their slots.
        if (indices && cu[seq] == cu[seq + 1]) {
            snapshot[i] = 0.f;
            continue;
        }
        const int slot = indices ? indices[seq] : seq;
        if (slot < 0 || slot >= slots) __trap();
        const int64_t base = (int64_t(slot) * heads + head) * 128 * 128;
        const int64_t src = base + (indices ? dv * 128 + dk : dk * 128 + dv);
        snapshot[i] = bf16 ? load(static_cast<const __ppu_bfloat16*>(initial)[src])
                           : static_cast<const float*>(initial)[src];
    }
}

template <typename E>
__device__ __forceinline__ double token_gate(const Arguments& a, int token, int head) {
    if (a.g) return double(a.g[int64_t(token) * a.heads + head]);
    // Re-evaluate the public FP32 raw-gating activation, never subtract nearby
    // cumulative gates. Q/K and beta already passed their public boundaries.
    const float x = load(static_cast<const E*>(a.raw_a)[
        int64_t(token) * a.a_token_stride + head]) + a.dt_bias[head];
    const float softplus = x <= 20.f ? log1pf(expf(x)) : x;
    return double(-expf(a.a_log[head]) * softplus);
}

template <typename E>
__device__ __forceinline__ void zero_suffix_inline(Arguments a) {
    const int last = a.cu[a.sequences];
    if (last < 0 || last > a.tokens) __trap();
    const int64_t begin = int64_t(last) * a.heads * 128;
    const int64_t end = int64_t(a.tokens) * a.heads * 128;
    for (int64_t i = begin + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < end; i += int64_t(gridDim.x) * blockDim.x)
        static_cast<E*>(a.out)[i] = convert<E>(0.f);
}

template <typename E>
__global__ void zero_suffix_kernel(Arguments a) {
    const int last = a.cu[a.sequences];
    if (last < 0 || last > a.tokens) __trap();
    const int64_t begin = int64_t(last) * a.heads * 128;
    const int64_t end = int64_t(a.tokens) * a.heads * 128;
    for (int64_t i = begin + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < end; i += int64_t(gridDim.x) * blockDim.x)
        static_cast<E*>(a.out)[i] = convert<E>(0.f);
}

// Each warp owns Columns independent DV columns and shares Q/K loads between
// them. Carry, reductions, and updates use the validated FP64 arithmetic.
template <typename E, int Columns, bool FuseSuffix = false, bool LoadIndexedState = false>
__global__ __launch_bounds__(128) void repair_kernel(Arguments a) {
    using Accumulator = double;
    if constexpr (FuseSuffix) zero_suffix_inline<E>(a);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int Groups = 32 / Columns;
    const int tasks = a.sequences * a.heads * Groups;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int dv_base = (task % Groups) * 4 * Columns + warp * Columns;
        const int sh = task / Groups, head = sh % a.heads, seq = sh / a.heads;
        const int kh = head / (a.heads / a.key_heads);
        if (a.flags && !a.flags[seq*a.key_heads+kh]) continue;
        const int bos = a.cu[seq], eos = a.cu[seq + 1];
        if (a.cu[0] != 0 || bos < 0 || eos < bos || eos > a.tokens) __trap();
        if (a.indexed && bos == eos) continue;
        int slot = 0;
        if constexpr (LoadIndexedState) {
            slot = a.indices[seq];
            if (slot < 0 || slot >= a.final_slots) __trap();
        }
        Accumulator state[Columns][4];
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                if constexpr (LoadIndexedState) {
                    // This CTA owns every key element of its DV columns for
                    // both the initial read and the final in-place store.
                    const int64_t offset = ((int64_t(slot) * a.heads + head) * 128 + dv_base+c) * 128 + lane + 32*j;
                    state[c][j] = a.state_bf16
                        ? Accumulator(load(static_cast<const __ppu_bfloat16*>(a.final_state)[offset]))
                        : Accumulator(static_cast<const float*>(a.final_state)[offset]);
                } else {
                    state[c][j] = a.snapshot
                        ? Accumulator(a.snapshot[(int64_t(sh) * 128 + dv_base+c) * 128 + lane + 32*j]) : 0.;
                }
        }
        for (int token = bos; token < eos; ++token) {
            Accumulator decay = 0., beta = 0.;
            if (lane == 0) {
                decay = exponential(Accumulator(token_gate<E>(a, token, head)));
                beta = Accumulator(a.beta[int64_t(token) * a.heads + head]);
            }
            decay = __shfl_sync(0xffffffff, decay, 0);
            beta = __shfl_sync(0xffffffff, beta, 0);
            Accumulator q[4], k[4], prediction[Columns] = {};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t off = (int64_t(token) * a.key_heads + kh) * 128 + lane + 32*j;
                q[j] = Accumulator(load(static_cast<const E*>(a.q)[off]));
                k[j] = Accumulator(load(static_cast<const E*>(a.k)[off]));
                #pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    state[c][j] *= decay;
                    prediction[c] = multiply_add(state[c][j], k[j], prediction[c]);
                }
            }
            Accumulator value[Columns], delta[Columns], output[Columns] = {};
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                prediction[c] = warp_sum(prediction[c]);
                Accumulator one = 0.;
                if (lane == 0) one = Accumulator(load(static_cast<const E*>(a.v)[
                    int64_t(token) * a.v_token_stride + head * 128 + dv_base+c]));
                value[c] = __shfl_sync(0xffffffff, one, 0);
                delta[c] = beta * (value[c] - prediction[c]);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int c = 0; c < Columns; ++c)
                    state[c][j] = multiply_add(delta[c], k[j], state[c][j]);
                #pragma unroll
                for (int c = 0; c < Columns; ++c)
                    output[c] = multiply_add(state[c][j], q[j], output[c]);
            }
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                output[c] = warp_sum(output[c]);
                if (lane == 0) static_cast<E*>(a.out)[
                    (int64_t(token) * a.heads + head) * 128 + dv_base+c] = convert<E>(float(output[c] * Accumulator(a.scale)));
            }
        }
        if (a.final_state) {
            const int slot = a.indexed ? a.indices[seq] : seq;
            if (slot < 0 || slot >= a.final_slots) __trap();
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int dk = lane + 32*j, dv = dv_base+c;
                    const int64_t off = (int64_t(slot) * a.heads + head) * 128 * 128
                        + (a.indexed ? dv * 128 + dk : dk * 128 + dv);
                    if (a.state_bf16) static_cast<__ppu_bfloat16*>(a.final_state)[off] = convert<__ppu_bfloat16>(float(state[c][j]));
                    else static_cast<float*>(a.final_state)[off] = float(state[c][j]);
                }
            }
        }
    }
}

// Offline staging variant. A CTA loads 32 tokens cooperatively, computes the
// same sequential recurrence, and publishes output with coalesced CTA stores.
// Each DV column retains the same dot-product and update order as repair_kernel.
template <typename E, int Columns, bool FuseSuffix = false>
__global__ __launch_bounds__(128) void repair_staged_kernel(Arguments a) {
    using Accumulator = double;
    if constexpr (FuseSuffix) zero_suffix_inline<E>(a);
    constexpr int Tile = 32, Width = 4 * Columns, Groups = 128 / Width;
    __shared__ E shared_q[Tile * 128], shared_k[Tile * 128];
    __shared__ E shared_v[Tile * Width], shared_out[Tile * Width];
    __shared__ Accumulator shared_decay[Tile], shared_beta[Tile];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tasks = a.sequences * a.heads * Groups;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int group_dv = (task % Groups) * Width;
        const int dv_base = group_dv + warp * Columns;
        const int sh = task / Groups, head = sh % a.heads, seq = sh / a.heads;
        const int kh = head / (a.heads / a.key_heads);
        if (a.flags && !a.flags[seq*a.key_heads+kh]) continue;
        const int bos = a.cu[seq], eos = a.cu[seq + 1];
        if (a.cu[0] != 0 || bos < 0 || eos < bos || eos > a.tokens) __trap();
        if (a.indexed && bos == eos) continue;
        Accumulator state[Columns][4];
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                state[c][j] = a.snapshot
                    ? Accumulator(a.snapshot[(int64_t(sh) * 128 + dv_base+c) * 128 + lane + 32*j]) : 0.;
        }
        for (int tile = bos; tile < eos; tile += Tile) {
            const int count = eos - tile < Tile ? eos - tile : Tile;
            for (int i = threadIdx.x; i < count * 128; i += 128) {
                const int token = tile + i / 128, dk = i % 128;
                const int64_t source = (int64_t(token) * a.key_heads + kh) * 128 + dk;
                shared_q[i] = static_cast<const E*>(a.q)[source];
                shared_k[i] = static_cast<const E*>(a.k)[source];
            }
            for (int i = threadIdx.x; i < count * Width; i += 128) {
                const int token = tile + i / Width, dv = group_dv + i % Width;
                shared_v[i] = static_cast<const E*>(a.v)[
                    int64_t(token) * a.v_token_stride + head * 128 + dv];
            }
            if (threadIdx.x < count) {
                const int token = tile + threadIdx.x;
                shared_decay[threadIdx.x] = exponential(Accumulator(token_gate<E>(a, token, head)));
                shared_beta[threadIdx.x] = Accumulator(a.beta[int64_t(token) * a.heads + head]);
            }
            __syncthreads();
            for (int token = 0; token < count; ++token) {
                const Accumulator decay = shared_decay[token], beta = shared_beta[token];
                Accumulator q[4], k[4], prediction[Columns] = {};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int offset = token * 128 + lane + 32*j;
                    q[j] = Accumulator(load(shared_q[offset]));
                    k[j] = Accumulator(load(shared_k[offset]));
                    #pragma unroll
                    for (int c = 0; c < Columns; ++c) {
                        state[c][j] *= decay;
                        prediction[c] = multiply_add(state[c][j], k[j], prediction[c]);
                    }
                }
                Accumulator value[Columns], delta[Columns], output[Columns] = {};
                #pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    prediction[c] = warp_sum(prediction[c]);
                    value[c] = Accumulator(load(shared_v[token * Width + warp * Columns+c]));
                    delta[c] = beta * (value[c] - prediction[c]);
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    #pragma unroll
                    for (int c = 0; c < Columns; ++c)
                        state[c][j] = multiply_add(delta[c], k[j], state[c][j]);
                    #pragma unroll
                    for (int c = 0; c < Columns; ++c)
                        output[c] = multiply_add(state[c][j], q[j], output[c]);
                }
                #pragma unroll
                for (int c = 0; c < Columns; ++c) {
                    output[c] = warp_sum(output[c]);
                    if (lane == 0) shared_out[token * Width + warp * Columns+c] =
                        convert<E>(float(output[c] * Accumulator(a.scale)));
                }
            }
            __syncthreads();
            for (int i = threadIdx.x; i < count * Width; i += 128) {
                const int token = tile + i / Width, dv = group_dv + i % Width;
                static_cast<E*>(a.out)[(int64_t(token) * a.heads + head) * 128 + dv] = shared_out[i];
            }
        }
        if (a.final_state) {
            const int slot = a.indexed ? a.indices[seq] : seq;
            if (slot < 0 || slot >= a.final_slots) __trap();
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int dk = lane + 32*j, dv = dv_base+c;
                    const int64_t off = (int64_t(slot) * a.heads + head) * 128 * 128
                        + (a.indexed ? dv * 128 + dk : dk * 128 + dv);
                    if (a.state_bf16) static_cast<__ppu_bfloat16*>(a.final_state)[off] = convert<__ppu_bfloat16>(float(state[c][j]));
                    else static_cast<float*>(a.final_state)[off] = float(state[c][j]);
                }
            }
        }
    }
}
}  // namespace

void launch_guard_snapshot_medium(const void* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, int sequences, int dtype_tag, bool include_empty,
    const float* beta, int heads, const void* initial, float* snapshot,
    const int32_t* indices, int slots, bool state_bf16, hggcStream_t stream) {
    const dim3 grid(key_heads * 4, sequences);
    if (heads == 8 && (sequences == 11 || sequences == 22)) {
        if (dtype_tag == 0)
            guard_snapshot_medium_kernel<__half,4><<<grid,128,0,stream>>>(
                static_cast<const __half*>(k),cu,flags,tokens,key_heads,include_empty,
                beta,heads,initial,snapshot,indices,slots,state_bf16);
        else
            guard_snapshot_medium_kernel<__ppu_bfloat16,4><<<grid,128,0,stream>>>(
                static_cast<const __ppu_bfloat16*>(k),cu,flags,tokens,key_heads,include_empty,
                beta,heads,initial,snapshot,indices,slots,state_bf16);
        return;
    }
    if (dtype_tag == 0)
        guard_snapshot_medium_kernel<__half,16><<<grid,512,0,stream>>>(
            static_cast<const __half*>(k),cu,flags,tokens,key_heads,include_empty,
            beta,heads,initial,snapshot,indices,slots,state_bf16);
    else
        guard_snapshot_medium_kernel<__ppu_bfloat16,16><<<grid,512,0,stream>>>(
            static_cast<const __ppu_bfloat16*>(k),cu,flags,tokens,key_heads,include_empty,
            beta,heads,initial,snapshot,indices,slots,state_bf16);
}

void launch_guard_snapshot_short(const void* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, int dtype_tag, bool include_empty,
    const float* beta, int heads, const void* initial, float* snapshot,
    const int32_t* indices, int slots, bool state_bf16, hggcStream_t stream) {
    if (dtype_tag == 0)
        guard_snapshot_short_kernel<<<256,128,0,stream>>>(
            static_cast<const __half*>(k),cu,flags,tokens,key_heads,include_empty,
            beta,heads,initial,snapshot,indices,slots,state_bf16);
    else
        guard_snapshot_short_kernel<<<256,128,0,stream>>>(
            static_cast<const __ppu_bfloat16*>(k),cu,flags,tokens,key_heads,include_empty,
            beta,heads,initial,snapshot,indices,slots,state_bf16);
}

void launch_snapshot(const void* initial, float* snapshot,
    const int32_t* indices, const int32_t* cu, int sequences, int heads, int slots,
    bool state_bf16, hggcStream_t stream, const int32_t* flags, int key_heads, bool allow_tiled) {
    const int64_t size = int64_t(sequences) * heads * 128 * 128;
    const int blocks = int((size + 255) / 256 < 4096 ? (size + 255) / 256 : 4096);
    if (allow_tiled && (sequences <= 3 || ((sequences == 16 || sequences == 32 || sequences == 64) && heads == 8)) && heads <= 32) {
        const int64_t tasks = int64_t(sequences) * heads * 16;
        const int grid = int(tasks < 4096 ? tasks : 4096);
        if (grid) snapshot_tiled_parallel_kernel<<<grid,256,0,stream>>>(
            initial,snapshot,indices,cu,sequences,heads,slots,state_bf16,flags,key_heads);
        return;
    }
    if (blocks) snapshot_kernel<<<blocks, 256, 0, stream>>>(
        initial, snapshot, indices, cu, sequences, heads, slots, state_bf16, flags, key_heads);
}

void launch_guard(const void* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, int sequences, int dtype_tag, bool include_empty,
    hggcStream_t stream, const float* beta, int heads, bool allow_parallel) {
    const int tasks = sequences*key_heads;
    if (!tasks) return;
    // Private current-890P experiment. Preserve the exact structural classifier,
    // short/empty rules and atomic OR result; expose independent windows only.
    if (allow_parallel && sequences >= 1 &&
        (sequences <= 3 || ((sequences == 16 || sequences == 32 || sequences == 64) && heads == 8 &&
         (key_heads == 2 || key_heads == 4))) && key_heads <= 16 &&
        ((tokens >= 1024 && tokens <= 32831) ||
         (sequences == 1 && heads == 16 && key_heads == 4 && tokens >= 768 && tokens < 1024))) {
        guard_init_parallel_kernel<<<1,256,0,stream>>>(
            cu,flags,tokens,key_heads,sequences,include_empty);
        const int windows=int((int64_t(tokens)+127)/128);
        if(dtype_tag==0)guard_windows_parallel_kernel<<<dim3(windows,key_heads),128,0,stream>>>(
            static_cast<const __half*>(k),cu,flags,tokens,key_heads,sequences,beta,heads);
        else guard_windows_parallel_kernel<<<dim3(windows,key_heads),128,0,stream>>>(
            static_cast<const __ppu_bfloat16*>(k),cu,flags,tokens,key_heads,sequences,beta,heads);
        return;
    }
    if (dtype_tag == 0)
        guard_kernel<<<tasks,128,0,stream>>>(static_cast<const __half*>(k),cu,flags,tokens,key_heads,include_empty,beta,heads);
    else
        guard_kernel<<<tasks,128,0,stream>>>(static_cast<const __ppu_bfloat16*>(k),cu,flags,tokens,key_heads,include_empty,beta,heads);
}

void launch_repair(Arguments a, int dtype_tag, hggcStream_t stream, int block_cap) {
    const int tasks = a.sequences * a.heads * (32 / (a.load_indexed_state ? 4 : a.dv_per_warp));
    if (!tasks) return;
    const int blocks = tasks < block_cap ? tasks : block_cap;
    if (a.load_indexed_state) {
        if (dtype_tag == 0)
            repair_kernel<__half, 4, true, true><<<blocks, 128, 0, stream>>>(a);
        else
            repair_kernel<__ppu_bfloat16, 4, true, true><<<blocks, 128, 0, stream>>>(a);
        return;
    }

#define LAUNCH_PRECISION_VARIANT(E, C) \
    do { \
        if (a.staged_tokens == 32) { \
            if (a.fuse_suffix) repair_staged_kernel<E, C, true><<<blocks, 128, 0, stream>>>(a); \
            else repair_staged_kernel<E, C><<<blocks, 128, 0, stream>>>(a); \
        } else { \
            if (a.fuse_suffix) repair_kernel<E, C, true><<<blocks, 128, 0, stream>>>(a); \
            else repair_kernel<E, C><<<blocks, 128, 0, stream>>>(a); \
        } \
    } while (0)
#define LAUNCH_PRECISION_REPAIR(E) \
    do { \
        if (!a.fuse_suffix) zero_suffix_kernel<E><<<256, 256, 0, stream>>>(a); \
        if (a.dv_per_warp == 4) LAUNCH_PRECISION_VARIANT(E, 4); \
        else if (a.dv_per_warp == 2) LAUNCH_PRECISION_VARIANT(E, 2); \
        else LAUNCH_PRECISION_VARIANT(E, 1); \
    } while (0)
    if (dtype_tag == 0) LAUNCH_PRECISION_REPAIR(__half);
    else LAUNCH_PRECISION_REPAIR(__ppu_bfloat16);
#undef LAUNCH_PRECISION_REPAIR
#undef LAUNCH_PRECISION_VARIANT
}
}  // namespace flashqla_precision
