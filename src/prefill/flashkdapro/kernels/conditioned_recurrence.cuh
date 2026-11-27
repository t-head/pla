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

#include "conditioned_repair.cuh"

namespace flashkda {
namespace conditioned {
// All tile flags are complete before this launch. Only selected heads copy
// their original state; normal heads perform no state traffic.
__global__ __launch_bounds__(128) void snapshot_kernel(
    ConditionedArguments a, const void* source, int total_tiles, int32_t* flags,
    float* snapshot) {
    const int seq = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    constexpr int width = 16;
    __shared__ int flagged, tile_begin, tile_count, warp_flags[4];
    __shared__ float initial_weight;
    if (tid == 0) {
        int tile_base = 0, tiles = (int64_t(a.tokens) + width - 1) / width;
        if (a.cu) {
            tiles = 0;
            if (a.cu[0] == 0) {
                if (a.sequences >= 64) {
                    const int* prefix = reinterpret_cast<const int*>(
                        static_cast<const BF16*>(source)
                        + int64_t(a.heads) * total_tiles * WS_SLOT_UNITS);
                    tile_base = prefix[seq];
                    tiles = prefix[seq + 1] - tile_base;
                } else {
                    for (int i = 0; i <= seq; ++i) {
                        const int bos = a.cu[i], eos = a.cu[i + 1];
                        if (bos < 0 || eos < bos || eos > a.tokens) {
                            tiles = 0;
                            break;
                        }
                        tiles = int((int64_t(eos) - bos + width - 1) / width);
                        if (i < seq) tile_base += tiles;
                    }
                }
            }
        }
        tile_begin = tile_base; tile_count = tiles;
    }
    __syncthreads();
    int local = 0;
    for (int i = tid; i < tile_count; i += 128) {
        const int64_t index = int64_t(head) * total_tiles + tile_begin + i;
        const BF16* tile = static_cast<const BF16*>(source) + index * WS_SLOT_UNITS;
        local |= *reinterpret_cast<const int32_t*>(tile + WS_OFF_CONDITION_U);
    }
    const int any = __any_sync(0xffffffff, local != 0);
    if ((tid & 31) == 0) warp_flags[tid >> 5] = any;
    __syncthreads();
    if (tid == 0) {
        flagged = warp_flags[0] | warp_flags[1] | warp_flags[2] | warp_flags[3];
        const int bos = a.cu ? a.cu[seq] : 0;
        const int eos = a.cu ? a.cu[seq + 1] : a.tokens;
        const int64_t length = int64_t(eos) - bos;
        // A padded/packed request can have three effective tokens even when the
        // physical tensor is longer. Protect its original carry before fast
        // writers, preserving the qualified FP32-activation/FP64 short path.
        if ((!a.cu || a.cu[0] == 0) && bos >= 0 && eos <= a.tokens
                && length > 0 && length <= 3)
            flagged = 1;
        flags[seq * a.heads + head] = flagged;
    }
    __syncthreads();
    if (a.value_guard) {
        const int group = tid / 4, lane4 = tid & 3;
        float scale = 0.f, largest_gate = -INFINITY;
        for (int i = lane4; i < tile_count; i += 4) {
            const int64_t index = int64_t(head) * total_tiles + tile_begin + i;
            const BF16* tile = static_cast<const BF16*>(source) + index * WS_SLOT_UNITS;
            const float value = reinterpret_cast<const float*>(tile + WS_OFF_VALUE_SCALE_U)[group];
            largest_gate = fmaxf(largest_gate,
                *reinterpret_cast<const float*>(tile + WS_OFF_VALUE_GATE_U));
            scale = fmaxf(scale, value);
        }
        float initial_square = 0.f;
        const int initial_slot = slot(a, seq);
        const bool valid = a.initial_state && initial_slot >= 0 && initial_slot < a.initial_slots;
        for (int i = lane4; i < 4 * 128; i += 4) {
            const int dv = group * 4 + i / 128, dk = i % 128;
            float value = 0.f;
            if (valid) {
                const int64_t off = ((int64_t(initial_slot) * a.heads + head) * 128 + dv) * 128 + dk;
                value = a.state_fp32 ? static_cast<const float*>(a.initial_state)[off]
                                    : float(static_cast<const BF16*>(a.initial_state)[off]);
            }
            initial_square = fmaf(value, value, initial_square);
        }
        #pragma unroll
        for (int mask = 2; mask; mask >>= 1) {
            initial_square += __shfl_xor_sync(0xffffffff, initial_square, mask, 4);
            scale = fmaxf(scale, __shfl_xor_sync(0xffffffff, scale, mask, 4));
            largest_gate = fmaxf(largest_gate,__shfl_xor_sync(0xffffffff,largest_gate,mask,4));
        }
        if (tid == 0) {
            const int length = a.cu ? a.cu[seq+1]-a.cu[seq] : a.tokens;
            initial_weight = length > 0 ? float(exp2(double(largest_gate)*length)) : 1.f;
            snapshot[(int64_t(seq) * a.heads + head) * 65 + 64] = largest_gate;
        }
        __syncthreads();
        if (lane4 == 0) {
            const int64_t base = (int64_t(seq) * a.heads + head) * 65 + group;
            snapshot[base] = fmaxf(scale,sqrtf(initial_square)*initial_weight);
            snapshot[base+32] = sqrtf(initial_square);
        }
        return;
    }
    const bool empty = a.cu ? a.cu[seq] == a.cu[seq+1] : a.tokens == 0;
    if ((!flagged && !empty) || !a.initial_state) return;
    preserve_initial_state(a, seq, head, snapshot, tid);
}

__global__ __launch_bounds__(128) void preserve_kernel(ConditionedArguments a, float* snapshot) {
    const int seq = blockIdx.x, head = blockIdx.y;
    preserve_initial_state(a, seq, head, snapshot, threadIdx.x);
}

// Keep the warp reduction and the rounded BF16 operands identical to repair.
__global__ __launch_bounds__(128) void select_repair_cache(
    ConditionedArguments a, const int32_t* flags, int32_t* owner, int capacity) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ int choices[4];
    int selected = 0x7fffffff;
    for (int sh = threadIdx.x; sh < a.sequences * a.heads; sh += 128) {
        if (!flags[sh]) continue;
        const int seq = sh / a.heads;
        const int length = a.cu ? a.cu[seq + 1] - a.cu[seq] : a.tokens;
        if (length >= 2048 && length <= capacity) selected = min(selected, sh);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        selected = min(selected, __shfl_xor_sync(0xffffffff, selected, offset));
    if (lane == 0) choices[warp] = selected;
    __syncthreads();
    if (threadIdx.x == 0) {
        owner[0] = min(min(choices[0], choices[1]), min(choices[2], choices[3]));
        owner[1] = 0;
    }
}

__global__ __launch_bounds__(128) void prepare_repair_tokens(
    ConditionedArguments a, const int32_t* owner, RepairToken* output) {
    const int sh = *owner;
    if (sh == 0x7fffffff) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int seq = sh / a.heads, head = sh % a.heads;
    const int bos = a.cu ? a.cu[seq] : 0;
    const int eos = a.cu ? a.cu[seq + 1] : a.tokens;
    const float a_exp32 = expf(a.a_log[head]);
    for (int token = bos + blockIdx.x * 4 + warp; token < eos; token += gridDim.x * 4) {
        RepairToken& cached = output[token - bos];
        float qnorm, knorm;
        repair_norm(a, token, head, lane, qnorm, knorm);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dk = lane + 32 * j;
            cached.q[dk] = repair_component(float(static_cast<const BF16*>(a.q)[input_activation(a.q_stride, token, head, dk)]),
                                            qnorm, a.fla_normalization);
            cached.k[dk] = repair_component(float(static_cast<const BF16*>(a.k)[input_activation(a.k_stride, token, head, dk)]),
                                            knorm, a.fla_normalization);
            cached.v[dk] = static_cast<const BF16*>(a.v)[input_activation(a.v_stride, token, head, dk)];
            cached.decay[dk] = repair_decay(a, activated_gate32(a, token, head, dk, a_exp32));
        }
        if (lane == 0) cached.beta = activated_beta32(a, token, head);
    }
}

constexpr int RepairWindow = 512;
constexpr int RepairOverlap = 256;
constexpr int RepairTile = 16;
constexpr int RepairMatrix = 128 * 128;

// Speculative overlap supplies only a candidate carry. The next kernel checks
// its complete FP64 bit pattern against the previous window's outgoing state.
// Window zero starts from the original state, so a successful check implies
// exact serial state at every boundary by induction.
__global__ __launch_bounds__(128) void repair_windows(
    ConditionedArguments a, const float* snapshot, const int32_t* owner,
    const RepairToken* prepared, double* incoming, double* outgoing) {
    const int sh = *owner;
    if (sh == 0x7fffffff) return;
    const int seq = sh / a.heads, head = sh % a.heads;
    const int bos = a.cu ? a.cu[seq] : 0;
    const int eos = a.cu ? a.cu[seq + 1] : a.tokens;
    const int segment = blockIdx.y;
    const int begin = bos + segment * RepairWindow;
    if (begin >= eos) return;
    const int end = min(begin + RepairWindow, eos);
    const int warm = max(bos, begin - RepairOverlap);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int dv = blockIdx.x * 4 + warp;
    const int state_slot = slot(a, seq);
    const int64_t boundary = int64_t(segment) * RepairMatrix + dv * 128 + lane;
    __shared__ __align__(8) unsigned char storage[RepairTile * sizeof(RepairToken)];
    auto* staged = reinterpret_cast<RepairToken*>(storage);
    double state[4] = {};
    if (warm == bos) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dk = lane + 32 * j;
            if (a.initial_state && snapshot) {
                state[j] = snapshot[(int64_t(sh) * 128 + dv) * 128 + dk];
            } else if (a.initial_state && state_slot >= 0 && state_slot < a.initial_slots) {
                const int64_t input = ((int64_t(state_slot) * a.heads + head) * 128 + dv) * 128 + dk;
                state[j] = a.state_fp32 ? static_cast<const float*>(a.initial_state)[input]
                                        : float(static_cast<const BF16*>(a.initial_state)[input]);
            }
        }
    }
    for (int first = warm; first < end; first += RepairTile) {
        const int last = min(first + RepairTile, end);
        const bool publish = first >= begin;
        if (first == begin && segment > 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) incoming[boundary + 32 * j] = state[j];
        }
        for (int i = 0; i < last - first; ++i) {
            const int dk = threadIdx.x;
            const auto& input = prepared[first - bos + i];
            staged[i].decay[dk] = input.decay[dk];
            staged[i].k[dk] = input.k[dk];
            staged[i].v[dk] = input.v[dk];
            if (publish) staged[i].q[dk] = input.q[dk];
            if (dk == 0) staged[i].beta = input.beta;
        }
        __syncthreads();
        for (int token = first; token < last; ++token) {
            const auto& x = staged[token - first];
            double key[4], q[4], pred = 0.;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dk = lane + 32 * j;
                key[j] = double(float(x.k[dk]));
                if (publish) q[j] = double(float(x.q[dk]));
                state[j] *= x.decay[dk];
                pred = fma(state[j], key[j], pred);
            }
            pred = warp_sum(pred);
            const double delta = double(x.beta) * (double(float(x.v[dv])) - pred);
            double out = 0.;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                state[j] = fma(delta, key[j], state[j]);
                if (publish) out = fma(state[j], q[j], out);
            }
            if (publish) {
                out = warp_sum(out);
                if (lane == 0)
                    static_cast<BF16*>(a.out)[activation(a, token, head, dv)] = BF16(float(out * a.scale));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        outgoing[boundary + 32 * j] = state[j];
        if (end == eos && a.final_state && state_slot >= 0 && state_slot < a.final_slots) {
            const int64_t i = ((int64_t(state_slot) * a.heads + head) * 128 + dv) * 128 + lane + 32 * j;
            if (a.state_fp32) static_cast<float*>(a.final_state)[i] = float(state[j]);
            else static_cast<BF16*>(a.final_state)[i] = BF16(float(state[j]));
        }
    }
}

__global__ __launch_bounds__(128) void check_repair_boundaries(
    ConditionedArguments a, const int32_t* owner, int32_t* mismatch,
    const double* incoming, const double* outgoing) {
    const int sh = *owner;
    if (sh == 0x7fffffff) return;
    const int seq = sh / a.heads;
    const int length = a.cu ? a.cu[seq + 1] - a.cu[seq] : a.tokens;
    const int count = ((length - 1) / RepairWindow) * RepairMatrix;
    bool wrong = false;
    for (int off = blockIdx.x * 128 + threadIdx.x; off < count; off += gridDim.x * 128) {
        const double prior = outgoing[off], next = incoming[off + RepairMatrix];
        wrong |= __double_as_longlong(prior) != __double_as_longlong(next);
    }
    const bool any = __any_sync(0xffffffff, wrong);
    if (any && (threadIdx.x & 31) == 0) atomicExch(mismatch, 1);
}

// A task owns four DV columns, with one warp per column and four DK values
// per lane. The persistent grid bounds normal unflagged launch overhead even
// for large batches; flagged tasks never read state already written by fast K2.
template <bool FloatState, bool MonitorValues = false, bool UseCache = false, bool LocalShort = false>
__global__ __launch_bounds__(128) void repair_kernel(
    ConditionedArguments a, const int32_t* flags, const float* snapshot,
    int32_t* output_flags, const int32_t* owner = nullptr,
    const RepairToken* prepared = nullptr) {
    using Acc = typename std::conditional<FloatState, float, double>::type;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float metric_sums[8];
    __shared__ int value_selected;
    const int tasks = a.sequences * a.heads * 32;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int dv_group = task % 32, sh = task / 32;
        const int head = sh % a.heads, seq = sh / a.heads;
        const int dv = dv_group * 4 + warp;
        const int bos = a.cu ? a.cu[seq] : 0;
        const int eos = a.cu ? a.cu[seq + 1] : a.tokens;
        if (bos == eos) {
            copy_empty_values(a, seq, head, dv_group * 4, 4, threadIdx.x, 128, snapshot);
            if (a.value_flags && threadIdx.x == 0)
                a.value_flags[int64_t(sh)*32+dv_group]=0;
            if (output_flags && dv_group == 0 && threadIdx.x == 0) output_flags[sh] = 0;
            continue;
        }
        const bool matrix_selected = !flags || flags[sh] != 0;
        bool selected = matrix_selected;
        if constexpr (MonitorValues) if (!selected && a.value_scales) {
            float state_square = 0.f;
            if (a.fast_state) {
                // Deferred legacy state is dense by request; an ordinary
                // caller-owned state buffer keeps the public slot mapping.
                const int fast_slot = a.publish_fast_state ? seq : slot(a, seq);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int64_t off = ((int64_t(fast_slot) * a.heads + head) * 128 + dv)
                                        * 128 + lane + 32*j;
                    const float value = fast_slot >= 0 && fast_slot < (a.publish_fast_state ? a.sequences : a.final_slots)
                        ? (a.state_fp32 ? static_cast<const float*>(a.fast_state)[off]
                                        : float(static_cast<const BF16*>(a.fast_state)[off])) : 0.f;
                    state_square = fmaf(value, value, state_square);
                }
            }
            const int feature = dv_group * 4 + (threadIdx.x & 3);
            const float scale = a.value_scales[int64_t(sh)*65+dv_group];
            const float initial_norm = a.value_scales[int64_t(sh)*65+32+dv_group];
            const float largest_gate = a.value_scales[int64_t(sh)*65+64];
            float initial_weight = exp2f(largest_gate*(threadIdx.x/4+1));
            const float initial_stride = exp2f(largest_gate*32.f);
            int output_risk = 0;
            for (int offset = 0; offset < (eos-bos+31)/32; ++offset) {
                const int token = bos+threadIdx.x/4+32*offset;
                const float value = token < eos
                    ? float(static_cast<const BF16*>(a.out)[activation(a,token,head,feature)]) : 0.f;
                float square = value*value;
                square += __shfl_xor_sync(0xffffffff,square,2,4);
                square += __shfl_xor_sync(0xffffffff,square,1,4);
                const float resolution = 0.0078125f*(scale+initial_norm*initial_weight)
                                         * fabsf(float(a.scale));
                if (token < eos)
                    output_risk |= !isfinite(square) || (resolution>0.f && sqrtf(square)<resolution);
                initial_weight *= initial_stride;
            }
            state_square = warp_sum(state_square);
            const int any_output_risk = __any_sync(0xffffffff,output_risk!=0);
            if (lane == 0) {
                metric_sums[warp] = state_square;
                metric_sums[4 + warp] = float(any_output_risk);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                const float ss = metric_sums[0] + metric_sums[1] + metric_sums[2] + metric_sums[3];
                const float os = metric_sums[4] + metric_sums[5] + metric_sums[6] + metric_sums[7];
                // Screen two BF16 units of relative resolution, allowing for
                // operand and intermediate rounding. This remains a heuristic,
                // not an advertised universal error certificate.
                const float resolution = 0.0078125f * scale;
                value_selected = !isfinite(ss) || !isfinite(os)
                    || (scale > 0.f && a.fast_state && sqrtf(ss) < resolution)
                    || os > 0.f;
            }
            __syncthreads();
            selected = value_selected != 0;
        }
        if constexpr (MonitorValues) if (a.value_flags && threadIdx.x == 0)
            a.value_flags[int64_t(sh) * 32 + dv_group] = matrix_selected ? 1 : (selected ? 2 : 0);
        if (!selected) {
            const int final_slot = slot(a, seq);
            if constexpr (MonitorValues) if (a.publish_fast_state && a.final_state && final_slot >= 0 && final_slot < a.final_slots) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int64_t src = (int64_t(sh) * 128 + dv) * 128 + lane + 32*j;
                    const int64_t dst = ((int64_t(final_slot) * a.heads + head) * 128 + dv) * 128 + lane + 32*j;
                    if (a.state_fp32) static_cast<float*>(a.final_state)[dst] = static_cast<const float*>(a.fast_state)[src];
                    else static_cast<BF16*>(a.final_state)[dst] = static_cast<const BF16*>(a.fast_state)[src];
                }
            }
            continue;
        }
        if (output_flags && dv_group == 0 && threadIdx.x == 0)
            output_flags[sh] = bos < eos ? 1 : 0;
        const float a_exp32 = expf(a.a_log[head]);
        Acc state[4];
        load_repair_state(a, state, seq, head, dv, lane, snapshot);
        const bool local_short = LocalShort && eos - bos <= KDA_DIRECT_MAX_TOKENS;
        const bool cached = local_short || (UseCache && sh == owner[0]);
        if constexpr (UseCache)
            if (!local_short && cached && owner[1] == 0) continue;
        constexpr int repair_tile = 16;
        __shared__ __align__(8) unsigned char storage[UseCache
            ? repair_tile * sizeof(RepairToken)
            : (LocalShort ? KDA_DIRECT_MAX_TOKENS * sizeof(RepairToken) : 1)];
        auto* staged = reinterpret_cast<RepairToken*>(storage);
        const int step = cached ? repair_tile : eos - bos;
        for (int first = bos; first < eos; first += step) {
            const int last = min(first + step, eos);
            if (cached) {
                for (int pos = 0; pos < last - first; ++pos) {
                    const int dk = threadIdx.x;
                    if (local_short) {
                        // Reuse gate/normalization work across the four value
                        // warps, retaining the original FP32/FP64 boundaries.
                        const int token = first + pos;
                        prepare_repair_token(a, staged[pos], token, head, dk, lane, a_exp32);
                    } else {
                        const auto& input = prepared[first - bos + pos];
                        staged[pos].decay[dk] = input.decay[dk];
                        staged[pos].q[dk] = input.q[dk];
                        staged[pos].k[dk] = input.k[dk];
                        staged[pos].v[dk] = input.v[dk];
                        if (dk == 0) staged[pos].beta = input.beta;
                    }
                }
                __syncthreads();
            }
            for (int token = first; token < last; ++token) {
                repair_step<FloatState>(a, state, cached ? staged + (token - first) : nullptr,
                                        token, head, dv, lane, a_exp32);
            }
            if (cached) __syncthreads();
        }
        store_repair_state(a, state, seq, head, dv, lane);
    }
}
}  // namespace conditioned

void launch_conditioned_workspace_snapshot(ConditionedArguments a, const void* ws,
                                             int total_tiles, int32_t* flags,
                                             float* snapshot, hggcStream_t stream) {
    if (a.sequences && a.heads)
        conditioned::snapshot_kernel<<<dim3(a.sequences, a.heads), 128, 0, stream>>>(
            a, ws, total_tiles, flags, snapshot);
}
void launch_conditioned_repair(ConditionedArguments a, const int32_t* flags,
                               const float* snapshot, hggcStream_t stream, bool cache_short_tokens) {
    const int tasks = a.sequences * a.heads * 32;
    if (tasks) {
        if (a.value_guard)
            conditioned::repair_kernel<false, true><<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, flags, snapshot, nullptr);
        else if (cache_short_tokens)
            conditioned::repair_kernel<false, false, false, true>
                <<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, flags, snapshot, nullptr);
        else
            conditioned::repair_kernel<false, false><<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, flags, snapshot, nullptr);
    }
}
int64_t conditioned_cache_bytes(int capacity) {
    const int windows = (capacity + conditioned::RepairWindow - 1) / conditioned::RepairWindow;
    return sizeof(int64_t) + int64_t(capacity) * sizeof(conditioned::RepairToken) +
        2 * int64_t(windows) * conditioned::RepairMatrix * sizeof(double);
}
void launch_conditioned_repair_cached(ConditionedArguments a, const int32_t* flags,
                                      const float* snapshot, void* workspace,
                                      int capacity, hggcStream_t stream) {
    auto* owner = static_cast<int32_t*>(workspace);
    auto* tokens = reinterpret_cast<conditioned::RepairToken*>(
        static_cast<char*>(workspace) + sizeof(int64_t));
    conditioned::select_repair_cache<<<1, 128, 0, stream>>>(a, flags, owner, capacity);
    conditioned::prepare_repair_tokens<<<(capacity + 3) / 4, 128, 0, stream>>>(a, owner, tokens);
    const int windows = (capacity + conditioned::RepairWindow - 1) / conditioned::RepairWindow;
    auto* incoming = reinterpret_cast<double*>(tokens + capacity);
    auto* outgoing = incoming + int64_t(windows) * conditioned::RepairMatrix;
    conditioned::repair_windows<<<dim3(32, windows), 128, 0, stream>>>(a, snapshot, owner, tokens,
                                                                      incoming, outgoing);
    conditioned::check_repair_boundaries<<<256, 128, 0, stream>>>(a, owner, owner + 1, incoming, outgoing);
    const int tasks = a.sequences * a.heads * 32;
    conditioned::repair_kernel<false, false, true>
        <<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, flags, snapshot, nullptr, owner, tokens);
}
void launch_conditioned_preserve(ConditionedArguments a, float* snapshot, hggcStream_t stream) {
    if (a.initial_state && a.sequences && a.heads)
        conditioned::preserve_kernel<<<dim3(a.sequences, a.heads), 128, 0, stream>>>(a, snapshot);
}
void launch_conditioned_direct(ConditionedArguments a, bool float_state, hggcStream_t stream,
                               int32_t* output_flags, const float* initial_snapshot) {
    const int tasks = a.sequences * a.heads * 32;
    if (!tasks) return;
    if (float_state)
        conditioned::repair_kernel<true, false><<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, nullptr, initial_snapshot, output_flags);
    else if (a.tokens <= KDA_DIRECT_MAX_TOKENS)
        conditioned::repair_kernel<false, false, false, true>
            <<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, nullptr, initial_snapshot, output_flags);
    else
        conditioned::repair_kernel<false, false><<<tasks < 256 ? tasks : 256, 128, 0, stream>>>(a, nullptr, initial_snapshot, output_flags);
}
}  // namespace flashkda
