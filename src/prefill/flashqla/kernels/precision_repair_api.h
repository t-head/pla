#pragma once

#include <hggc_runtime.h>
#include <cstdint>

namespace flashqla_precision {
struct Arguments {
    const void *q, *k, *v, *raw_a, *live_initial;
    const float *g, *beta, *a_log, *dt_bias, *snapshot;
    const int32_t *cu, *indices;
    const int32_t* flags;
    void *out, *final_state;
    int tokens, heads, key_heads, sequences, final_slots, initial_slots;
    int dv_per_warp;
    int staged_tokens;
    int64_t v_token_stride, a_token_stride;
    double scale;
    bool indexed, state_bf16, initial_bf16;
    bool fuse_suffix, load_indexed_state;
};

void launch_guard_snapshot_short(const void* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, int dtype_tag, bool include_empty,
    const float* beta, int heads, const void* initial, float* snapshot,
    const int32_t* indices, int slots, bool state_bf16, hggcStream_t stream);
void launch_guard_snapshot_medium(const void* k, const int32_t* cu, int32_t* flags,
    int tokens, int key_heads, int sequences, int dtype_tag, bool include_empty,
    const float* beta, int heads, const void* initial, float* snapshot,
    const int32_t* indices, int slots, bool state_bf16, hggcStream_t stream);
void launch_snapshot(const void* initial, float* snapshot,
                     const int32_t* indices, const int32_t* cu, int sequences, int heads,
                     int slots, bool state_bf16, hggcStream_t stream,
                     const int32_t* flags = nullptr, int key_heads = 0, bool allow_tiled = false);
void launch_guard(const void* k, const int32_t* cu, int32_t* flags,
                  int tokens, int key_heads, int sequences, int dtype_tag,
                  bool include_empty, hggcStream_t stream,
                  const float* beta = nullptr, int heads = 0, bool allow_parallel = false);
void launch_repair(Arguments args, int dtype_tag, hggcStream_t stream, int block_cap = 256);
}  // namespace flashqla_precision
