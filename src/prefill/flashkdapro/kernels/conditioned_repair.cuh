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

#pragma once

#include <type_traits>
#include <cutlass/bfloat16.h>
#include "prefill/flashkdapro/common/conditioned_api.h"
#include "flashkda_ppu_util.cuh"

namespace flashkda {
namespace conditioned {
using BF16 = cutlass::bfloat16_t;

__device__ __forceinline__ double warp_sum(double value) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}
__device__ __forceinline__ int slot(const ConditionedArguments& a, int seq) {
    return a.indices ? a.indices[seq] : seq;
}
__device__ __forceinline__ int64_t activation(
    const ConditionedArguments& a, int token, int head, int dk) {
    return (int64_t(token) * a.heads + head) * 128 + dk;
}
// Q/K/V can be aligned views over fused projections. Gate/output remain dense.
__device__ __forceinline__ int64_t input_activation(
    int64_t stride, int token, int head, int feature) {
    return int64_t(token) * stride + head * 128 + feature;
}
__device__ __forceinline__ float activated_beta32(
    const ConditionedArguments& a, int token, int head) {
    const float raw = float(static_cast<const BF16*>(a.beta)[
        int64_t(token) * a.beta_stride + head]);
    return a.fla_normalization ? 1.f / (1.f + expf(-raw))
                              : flashkda::sigmoid_tanh_approx_f32(raw);
}
__device__ __forceinline__ float activated_gate32(
    const ConditionedArguments& a, int token, int head, int dk, float a_exp) {
    if (a.lower_bound == 0.) return 0.f;
    const float raw = float(static_cast<const BF16*>(a.g)[activation(a, token, head, dk)]);
    const float z = a_exp * (raw + a.dt_bias[head * 128 + dk]);
    return a.fla_normalization
        ? float(a.lower_bound) / (1.f + expf(-z))
        : float(a.lower_bound * 1.4426950408889634) *
            flashkda::sigmoid_tanh_approx_f32(z);
}
__device__ __forceinline__ float decay32(const ConditionedArguments& a, float x) {
    return a.fla_normalization ? expf(x) : flashkda::ex2_approx_ftz_f32(x);
}
__device__ __forceinline__ void repair_norm(ConditionedArguments a, int token,
                                            int head, int lane, float& qnorm, float& knorm) {
    float qs = 0.f, ks = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int dk = (lane & 15) * 8 + j;
        const float qv = float(static_cast<const BF16*>(a.q)[input_activation(a.q_stride, token, head, dk)]);
        const float kv = float(static_cast<const BF16*>(a.k)[input_activation(a.k_stride, token, head, dk)]);
        qs += qv * qv; ks += kv * kv;
    }
    #pragma unroll
    for (int mask = 8; mask; mask >>= 1) {
        qs += __shfl_xor_sync(0xffffffff, qs, mask, 16);
        ks += __shfl_xor_sync(0xffffffff, ks, mask, 16);
    }
    qnorm = a.fla_normalization ? sqrtf(qs + 1e-6f) : rsqrtf(qs + 1e-6f);
    knorm = a.fla_normalization ? sqrtf(ks + 1e-6f) : rsqrtf(ks + 1e-6f);
}

__device__ __forceinline__ BF16 repair_component(float x, float norm, bool fla) {
    return BF16(fla ? x / norm : x * norm);
}

__device__ __forceinline__ double repair_decay(ConditionedArguments a, float value) {
    return a.fla_normalization ? exp(double(value)) : exp2(double(value));
}

struct RepairToken {
    double decay[128];
    BF16 q[128], k[128], v[128];
    float beta;
};

__device__ __forceinline__ void prepare_repair_token(
    const ConditionedArguments& a, RepairToken& staged, int token,
    int head, int dk, int lane, float a_exp32) {
    float qnorm, knorm;
    repair_norm(a, token, head, lane, qnorm, knorm);
    staged.q[dk] = repair_component(float(static_cast<const BF16*>(a.q)[
        input_activation(a.q_stride, token, head, dk)]), qnorm, a.fla_normalization);
    staged.k[dk] = repair_component(float(static_cast<const BF16*>(a.k)[
        input_activation(a.k_stride, token, head, dk)]), knorm, a.fla_normalization);
    staged.v[dk] = static_cast<const BF16*>(a.v)[input_activation(a.v_stride, token, head, dk)];
    const float gate = activated_gate32(a, token, head, dk, a_exp32);
    staged.decay[dk] = repair_decay(a, gate);
    if (dk == 0) staged.beta = activated_beta32(a, token, head);
}

template <class Acc>
__device__ __forceinline__ void load_repair_state(
    const ConditionedArguments& a, Acc (&state)[4], int seq, int head,
    int dv, int lane, const float* snapshot) {
    const int sh = seq * a.heads + head;
    const int initial_slot = slot(a, seq);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        state[j] = Acc(0);
        if (a.initial_state) {
            if (snapshot) state[j] = Acc(snapshot[(int64_t(sh) * 128 + dv) * 128 + lane + 32*j]);
            else if (initial_slot >= 0 && initial_slot < a.initial_slots) {
                const int64_t off = ((int64_t(initial_slot) * a.heads + head) * 128 + dv)
                                  * 128 + lane + 32*j;
                state[j] = a.state_fp32 ? Acc(static_cast<const float*>(a.initial_state)[off])
                                       : Acc(float(static_cast<const BF16*>(a.initial_state)[off]));
            }
        }
    }
}

template <class Acc>
__device__ __forceinline__ void store_repair_state(
    const ConditionedArguments& a, const Acc (&state)[4], int seq, int head, int dv, int lane) {
    const int state_slot = slot(a, seq);
    if (a.final_state && state_slot >= 0 && state_slot < a.final_slots) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t off = ((int64_t(state_slot) * a.heads + head) * 128 + dv)
                              * 128 + lane + 32*j;
            if (a.state_fp32) static_cast<float*>(a.final_state)[off] = float(state[j]);
            else static_cast<BF16*>(a.final_state)[off] = BF16(float(state[j]));
        }
    }
}

template <bool FloatState, bool Cached = false>
__device__ __forceinline__ void repair_update(
    const ConditionedArguments& a,
    typename std::conditional<FloatState, float, double>::type (&state)[4],
    const typename std::conditional<FloatState, float, double>::type (&q)[4],
    const typename std::conditional<FloatState, float, double>::type (&k)[4],
    const RepairToken* input,
    int token, int head, int dv, int lane) {
    using Acc = typename std::conditional<FloatState, float, double>::type;
    Acc prediction = Acc(0);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (FloatState) prediction = fmaf(state[j], k[j], prediction);
        else prediction = fma(state[j], k[j], prediction);
    }
    prediction = warp_sum(prediction);
    const Acc b = Acc((Cached || input) ? input->beta : activated_beta32(a, token, head));
    const Acc value = Acc(float((Cached || input) ? input->v[dv]
        : static_cast<const BF16*>(a.v)[input_activation(a.v_stride, token, head, dv)]));
    const Acc delta = b * (value - prediction);
    Acc output = Acc(0);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (FloatState) {
            state[j] = fmaf(delta, k[j], state[j]);
            output = fmaf(state[j], q[j], output);
        } else {
            state[j] = fma(delta, k[j], state[j]);
            output = fma(state[j], q[j], output);
        }
    }
    output = warp_sum(output);
    if (lane == 0) static_cast<BF16*>(a.out)[activation(a, token, head, dv)] =
        BF16(float(output * a.scale));

}

template <bool FloatState>
__device__ __forceinline__ void repair_step(
    const ConditionedArguments& a,
    typename std::conditional<FloatState, float, double>::type (&state)[4],
    const RepairToken* input, int token, int head, int dv, int lane, float a_exp32) {
    using Acc = typename std::conditional<FloatState, float, double>::type;
    float qnorm = 0.f, knorm = 0.f;
    if (!input) repair_norm(a, token, head, lane, qnorm, knorm);
    Acc q[4], k[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dk = lane + 32*j;
        if (input) {
            const auto& load = *input;
            q[j] = Acc(float(load.q[dk]));
            k[j] = Acc(float(load.k[dk]));
            state[j] *= load.decay[dk];
        } else {
            const float qv = float(static_cast<const BF16*>(a.q)[input_activation(a.q_stride, token, head, dk)]);
            const float kv = float(static_cast<const BF16*>(a.k)[input_activation(a.k_stride, token, head, dk)]);
            q[j] = Acc(float(repair_component(qv, qnorm, a.fla_normalization)));
            k[j] = Acc(float(repair_component(kv, knorm, a.fla_normalization)));
            const float activated = activated_gate32(a, token, head, dk, a_exp32);
            if constexpr (FloatState) state[j] *= decay32(a, activated);
            else state[j] *= repair_decay(a, activated);
        }
    }
    repair_update<FloatState>(a, state, q, k, input, token, head, dv, lane);

}

// One warp owns one value row; all arithmetic is shared with the
// standalone precision kernel, including its FP32 activation boundaries.
// A positive index denotes a strictly interior, sequence-local 64-token boundary.
__device__ __forceinline__ int track_token(const TrackState& track, int seq, int length) {
    const int64_t chunk = track.int64 ? static_cast<const int64_t*>(track.chunk_idx)[seq]
                                    : static_cast<const int32_t*>(track.chunk_idx)[seq];
    return chunk > 0 && chunk <= (length - 1) / 64 ? int(chunk * 64) : -1;
}

template <class Acc = double, bool Track = false>
__device__ __forceinline__ void repair_value(
    const ConditionedArguments& a, int seq, int head, int dv, int lane,
    const RepairToken* prepared = nullptr, const TrackState* track = nullptr) {
    const int bos = a.cu ? a.cu[seq] : 0;
    const int eos = a.cu ? a.cu[seq + 1] : a.tokens;
    if (bos == eos) {
        // Identity preserves the public dtype, including NaN payload bits.
        const int state_slot = slot(a, seq);
        if (a.final_state && state_slot >= 0 && state_slot < a.final_slots) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t offset = ((int64_t(state_slot) * a.heads + head) * 128 + dv)
                    * 128 + lane + 32*j;
                const bool has_initial = a.initial_state && state_slot < a.initial_slots;
                if (a.state_fp32)
                    static_cast<float*>(a.final_state)[offset] = has_initial
                        ? static_cast<const float*>(a.initial_state)[offset] : 0.f;
                else
                    static_cast<BF16*>(a.final_state)[offset] = has_initial
                        ? static_cast<const BF16*>(a.initial_state)[offset] : BF16(0.f);
            }
        }
        return;
    }
    Acc state[4];
    load_repair_state(a, state, seq, head, dv, lane, nullptr);
    const float a_exp32 = expf(a.a_log[head]);
    const int boundary = Track ? track_token(*track, seq, eos - bos) : -1;
    for (int token = bos; token < eos; ++token) {
        repair_step<std::is_same<Acc, float>::value>(a, state, prepared ? prepared + token - bos : nullptr,
                           token, head, dv, lane, a_exp32);
        if constexpr (Track) {
            if (token - bos + 1 == boundary) {
                const int64_t base = ((int64_t(seq) * a.heads + head) * 128 + dv) * 128;
                #pragma unroll
                for (int j = 0; j < 4; ++j) track->state[base + lane + 32*j] = float(state[j]);
            }
        }
    }
    store_repair_state(a, state, seq, head, dv, lane);
}

// BF16 embeds exactly in the high FP32 word. Preserve that representation
// without a numeric conversion, including NaN payloads needed by empty rows.
__device__ __forceinline__ void preserve_initial_state(
    const ConditionedArguments& a, int seq, int head, float* snapshot, int tid) {
    const int state_slot = slot(a, seq);
    const bool valid = a.initial_state && state_slot >= 0 && state_slot < a.initial_slots;
    const int64_t src = (int64_t(state_slot) * a.heads + head) * 128 * 128;
    const int64_t dst = (int64_t(seq) * a.heads + head) * 128 * 128;
    for (int i = tid; i < 128 * 128; i += 128) {
        uint32_t bits = 0;
        if (valid) bits = a.state_fp32 ? static_cast<const uint32_t*>(a.initial_state)[src + i]
            : uint32_t(static_cast<const uint16_t*>(a.initial_state)[src + i]) << 16;
        reinterpret_cast<uint32_t*>(snapshot)[dst + i] = bits;
    }
}

// Empty requests are identities. Copy contiguous public-state packets once
// per CTA, retaining raw dtype bits and masking invalid state-pool slots.
__device__ __forceinline__ void copy_empty_values(
    const ConditionedArguments& a, int seq, int head, int first_value,
    int rows, int tid, int threads, const float* snapshot = nullptr) {
    const int state_slot = slot(a, seq);
    if (!a.final_state || state_slot < 0 || state_slot >= a.final_slots) return;
    const int64_t offset = ((int64_t(state_slot) * a.heads + head) * 128 + first_value) * 128;
    const int bytes_per_element = a.state_fp32 ? 4 : 2;
    auto* destination = static_cast<unsigned char*>(a.final_state) + offset * bytes_per_element;
    if (snapshot) {
        const auto* saved = reinterpret_cast<const uint32_t*>(snapshot)
            + ((int64_t(seq) * a.heads + head) * 128 + first_value) * 128;
        for (int i = tid; i < rows * 128; i += threads) {
            if (a.state_fp32) reinterpret_cast<uint32_t*>(destination)[i] = saved[i];
            else reinterpret_cast<uint16_t*>(destination)[i] = uint16_t(saved[i] >> 16);
        }
        return;
    }
    const auto* source = a.initial_state && state_slot < a.initial_slots
        ? static_cast<const unsigned char*>(a.initial_state) + offset * bytes_per_element : nullptr;
    const bool vectorized = (reinterpret_cast<uintptr_t>(destination) & 15) == 0
        && (!source || (reinterpret_cast<uintptr_t>(source) & 15) == 0);
    if (vectorized) {
        for (int packet = tid; packet < rows * 128 * bytes_per_element / 16; packet += threads)
            reinterpret_cast<uint4*>(destination)[packet] = source
                ? reinterpret_cast<const uint4*>(source)[packet] : make_uint4(0, 0, 0, 0);
    } else if (a.state_fp32) {
        for (int i = tid; i < rows * 128; i += threads)
            reinterpret_cast<uint32_t*>(destination)[i] = source
                ? reinterpret_cast<const uint32_t*>(source)[i] : 0;
    } else {
        for (int i = tid; i < rows * 128; i += threads)
            reinterpret_cast<uint16_t*>(destination)[i] = source
                ? reinterpret_cast<const uint16_t*>(source)[i] : 0;
    }
}

// Independent value rows share cached inputs while retaining each row's
// exact operation order. Batch state I/O before stepping the rows to expose
// independent memory and arithmetic work in short precision requests.
template <class Acc>
__device__ __forceinline__ Acc repair_accumulate(Acc x, Acc y, Acc sum) {
    if constexpr (std::is_same<Acc, float>::value) return fmaf(x, y, sum);
    else return fma(x, y, sum);
}

template <int Values, class Acc = double>
__device__ __forceinline__ void repair_cached_values(
    const ConditionedArguments& a, int seq, int head, int first_value,
    int lane, const RepairToken* prepared) {
    const int bos = a.cu ? a.cu[seq] : 0;
    const int eos = a.cu ? a.cu[seq + 1] : a.tokens;
    if (bos == eos) {
        #pragma unroll
        for (int r = 0; r < Values; ++r)
            repair_value(a, seq, head, first_value + r, lane);
        return;
    }
    Acc state[Values][4];
    #pragma unroll
    for (int r = 0; r < Values; ++r)
        load_repair_state(a, state[r], seq, head, first_value + r, lane, nullptr);
    for (int token = bos; token < eos; ++token) {
        const auto& input = prepared[token - bos];
        Acc q[4], k[4], decay[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dk = lane + 32*j;
            q[j] = Acc(float(input.q[dk]));
            k[j] = Acc(float(input.k[dk]));
            decay[j] = Acc(input.decay[dk]);
        }
        // Independent value rows share the same inputs. Interleave each
        // reduction level across rows to expose independent instruction work;
        // every row retains the original FMA and XOR-reduction order.
        Acc work[Values], projected[Values];
        #pragma unroll
        for (int r = 0; r < Values; ++r) {
            work[r] = 0.;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                state[r][j] *= decay[j];
                work[r] = repair_accumulate(state[r][j], k[j], work[r]);
            }
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            #pragma unroll
            for (int r = 0; r < Values; ++r)
                work[r] += __shfl_xor_sync(0xffffffff, work[r], offset);
        }
        #pragma unroll
        for (int r = 0; r < Values; ++r) {
            const Acc value = Acc(float(input.v[first_value + r]));
            work[r] = Acc(input.beta) * (value - work[r]);
            projected[r] = 0.;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                state[r][j] = repair_accumulate(work[r], k[j], state[r][j]);
                projected[r] = repair_accumulate(state[r][j], q[j], projected[r]);
            }
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            #pragma unroll
            for (int r = 0; r < Values; ++r)
                projected[r] += __shfl_xor_sync(0xffffffff, projected[r], offset);
        }
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < Values; ++r)
                static_cast<BF16*>(a.out)[activation(a, token, head, first_value + r)] =
                    BF16(float(projected[r] * Acc(a.scale)));
        }
    }
    #pragma unroll
    for (int r = 0; r < Values; ++r)
        store_repair_state(a, state[r], seq, head, first_value + r, lane);
}

} // namespace conditioned
} // namespace flashkda

namespace flashkda {
namespace conditioned {
// Normalized BF16 geometry selects conservative parallel, disjoint-support,
// or roundoff-certified orthogonal classes. The FP32 rejection is certified
// against the stated reduction error; uncertain pairs use exact dyadic FP64
// products. This structural screen is not a universal operator error bound.
template <class Key, int Stride = 128>
__device__ __forceinline__ bool exact_structural_pair(
    const Key* keys, int count, int lane) {
    if (count < 2) return false;
    float af[4], bf[4], dot32 = 0.f, absolute32 = 0.f;
    bool exact_products = true;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dk = lane+32*j;
        af[j] = float(keys[dk]);
        bf[j] = float(keys[Stride+dk]);
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
    // Normal products are exact (<=16 significant bits). Four local terms
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
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        dot64 += __shfl_xor_sync(0xffffffff,dot64,offset);
        absolute64 += __shfl_xor_sync(0xffffffff,absolute64,offset);
    }
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

} // namespace conditioned
} // namespace flashkda
