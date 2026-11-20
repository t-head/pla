// SPDX-License-Identifier: Apache-2.0
using BF = __ppu_bfloat16;
using HF = __half;
constexpr int THREADS = 128;

template <class TT>
__device__ __forceinline__ int64_t index(const TT& t, int64_t i) {
  i *= t.stride[0];
  if (t.dtype == I64) return static_cast<const int64_t*>(t.data)[i];
  if (t.dtype == I8) return static_cast<const int8_t*>(t.data)[i];
  return static_cast<const int32_t*>(t.data)[i];
}

// Metadata known on the host to be int32 is read directly; otherwise the dynamic path is used.
template <bool I32, class TT>
__device__ __forceinline__ int64_t meta_index(const TT& t, int64_t i) {
  if constexpr (I32) return static_cast<const int32_t*>(t.data)[i * t.stride[0]];
  else return index(t, i);
}

// Compact launch parameters: full ABI tensors carry six strides (64 bytes each).
struct LiteTensor {
  void* data;
  int64_t stride[2];
  DType dtype;
};

inline LiteTensor lite(const Tensor& t) { return {t.data, {t.stride[0], t.stride[1]}, t.dtype}; }

__device__ __forceinline__ int bounded_index(const Tensor& t, int64_t row, int low, int high) {
  const int64_t value = index(t, row);
  assert(value >= low && value <= high);
  return static_cast<int>(value);
}

__device__ __forceinline__ void put_index(const Tensor& t, int64_t i, int64_t x) {
  i *= t.stride[0];
  if (t.dtype == I64) static_cast<int64_t*>(t.data)[i] = x;
  else if (t.dtype == I8) static_cast<int8_t*>(t.data)[i] = x;
  else static_cast<int32_t*>(t.data)[i] = x;
}

template <class T, class TT>
__device__ __forceinline__ float read(const TT& t, int64_t i) {
  return static_cast<float>(static_cast<const T*>(t.data)[i]);
}

template <class T, class TT>
__device__ __forceinline__ void write(const TT& t, int64_t i, float x) {
  static_cast<T*>(t.data)[i] = static_cast<T>(x);
}

template <class TT>
__device__ __forceinline__ float read_any(const TT& t, int64_t i) {
  if (t.dtype == BF16) return read<BF>(t, i);
  if (t.dtype == F16) return read<HF>(t, i);
  return read<float>(t, i);
}

// Unconditional vector loads; callers guarantee alignment on the host (see aligned_rows).
__device__ __forceinline__ void load_bf16x8(const void* src, float (&out)[8]) {
  const uint4 raw = *static_cast<const uint4*>(src);
  const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void load_bf16x4(const void* src, float (&out)[4]) {
  const uint2 raw = *static_cast<const uint2*>(src);
  out[0] = __uint_as_float(raw.x << 16);
  out[1] = __uint_as_float(raw.x & 0xffff0000u);
  out[2] = __uint_as_float(raw.y << 16);
  out[3] = __uint_as_float(raw.y & 0xffff0000u);
}

// Reads N contiguous elements with one 8- or 16-byte load when aligned and complete.
template <class T, int N, class View>
__device__ __forceinline__ void read_vector(const View& t, int64_t offset, int valid, float (&out)[N]) {
  constexpr int BYTES = N * sizeof(T);
  static_assert(BYTES == 8 || BYTES == 16, "vector loads are 8 or 16 bytes");
  const T* src = static_cast<const T*>(t.data) + offset;
  if (valid == N && (reinterpret_cast<uintptr_t>(src) & (BYTES - 1)) == 0) {
    using Raw = typename std::conditional<BYTES == 16, uint4, uint2>::type;
    const Raw raw = *reinterpret_cast<const Raw*>(src);
    const T* elements = reinterpret_cast<const T*>(&raw);
    #pragma unroll
    for (int i = 0; i < N; ++i) out[i] = static_cast<float>(elements[i]);
  } else {
    #pragma unroll
    for (int i = 0; i < N; ++i) out[i] = i < valid ? static_cast<float>(src[i]) : 0.f;
  }
}

template <class T>
__device__ __forceinline__ float rounded(float x) {
  return static_cast<float>(static_cast<T>(x));
}

__device__ __forceinline__ float operand(float x, bool tf32) {
  if (!tf32) return x;
  unsigned bits = __float_as_uint(x);
  if ((bits & 0x7f800000u) == 0x7f800000u) return x;
  // tc02 TF32 operands truncate the mantissa, rather than rounding to nearest.
  return __uint_as_float(bits & 0xffffe000u);
}

__device__ __forceinline__ float sum_warp(float x) {
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

__device__ __forceinline__ float sum_row(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x;
}

__device__ __forceinline__ float norm_denominator(float square_sum, int policy, float epsilon) {
  return policy == 0 ? 1.f : policy == 1 ? sqrtf(square_sum + epsilon) : sqrtf(square_sum) + epsilon;
}
