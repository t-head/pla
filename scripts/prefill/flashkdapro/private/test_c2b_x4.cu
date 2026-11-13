// Standalone tc02 C-to-B raw-bit regression for the production helper.
// Run the following from the repository root, using the same Actlize headers
// as the FlashKDA build (replace /task/actlize if installed elsewhere):
//   /usr/local/PPU_SDK/bin/hgcc -O3 -std=c++20 --use_fast_math \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -arch=ppu_15 -DUSE_PPU -DCUTE_USE_PACKED_TUPLE=1 \
//     -Isrc -I/task/actlize/include -I/task/actlize/tools/util/include \
//     -I/usr/local/PPU_SDK/targets/x86_64-linux/include \
//     scripts/prefill/flashkdapro/private/test_c2b_x4.cu -o /tmp/test_c2b_x4
// Map an authorized tc02 device to logical device 0 before running:
//   /tmp/test_c2b_x4                  # all modes at NT32/64/128/256
//   /tmp/test_c2b_x4 --threads 128     # all modes at NT128
//   /tmp/test_c2b_x4 --threads both    # NT128/256 compatibility selection
//   /tmp/test_c2b_x4 --smoke           # tags + rounded ACC; not exhaustive
// Default qualification visits every uint16 bit pattern at every source
// position. It compares the production x4 helper with explicit legacy x1
// loads, the SHFL route, and an independent scalar CPU mapping. This is a
// bit-mapping regression, not an end-to-end accuracy or performance benchmark.

#include <hggc_runtime.h>
#include "prefill/flashkdapro/kernels/flashkda_ppu_util.cuh"

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ != 150
#error "test_c2b_x4 requires the tc02 target: -arch=ppu_15"
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kWarp = 32;
constexpr int kRegs = 4;
constexpr int kGuard = 32;
constexpr int kScratchWords = 128;
constexpr int kWarpStride = kGuard + kScratchWords + kGuard;
constexpr uint32_t kGlobalCanary = 0xa5a5a5a5u;
constexpr int kRoutes = 4;  // shared pack, legacy x1, production x4, SHFL

enum Mode { kExhaustive = 0, kTagged = 1, kRoundedAcc = 2 };

const char* mode_name(Mode mode) {
  return mode == kExhaustive ? "all_u16_each_source_position"
       : mode == kTagged ? "tagged_words" : "rounded_acc_same_pack";
}

// Includes NaN/Inf/signed zero, f32 subnormals, BF16 halfway cases and overflow.
// For ACC we DO NOT predict the compiler's NaN payload conversion on the CPU:
// the device packs once and routes those identical packed bits to all converters.
__device__ __constant__ uint32_t kAccFloatBits[32] = {
    0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
    0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7f800001u,
    0xffc00123u, 0x00800000u, 0x007fffffu, 0x00000001u,
    0x80000001u, 0x3f808000u, 0x3f818000u, 0x3f807fffu,
    0x3f808001u, 0x7f7fffffu, 0x7f7f0000u, 0xff7fffffu,
    0x00808000u, 0x807fffffu, 0x3eaaaaabu, 0xbeaaaaabu,
    0x40008000u, 0xc0008000u, 0x00008000u, 0x80008000u,
    0x3f7fffffu, 0x3f800001u, 0x7fcabcdeu, 0xff812345u};

__host__ __device__ uint16_t raw_half(Mode mode, uint32_t seed, int position) {
  if (mode == kExhaustive) {
    // For EVERY fixed position in [0,256), seed=0..65535 visits all 65536
    // uint16_t patterns exactly once. 257 is odd, so all positions within a
    // warp also have distinct tags. No floating conversion occurs here.
    return static_cast<uint16_t>(seed + uint32_t(position) * 257u);
  }
  return static_cast<uint16_t>(0x8001u + seed * 256u + uint32_t(position));
}

__device__ __forceinline__ uint32_t scratch_canary(uint32_t seed, int lane,
                                                 bool suffix) {
  return (suffix ? 0x2468ace0u : 0x13579bdfu) ^ (seed * 0x9e3779b9u)
         ^ uint32_t(lane);
}

// Freeze the previous batched implementation here. It must remain four
// explicit x1 instructions rather than calling the production batched helper.
// This keeps the reference independent when that helper uses x4.
__device__ __forceinline__ void legacy_batched_x1(
    const uint32_t c[4], uint32_t b[4], int lane, uint32_t* scratch) {
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) scratch[i * 32 + lane] = c[i];
  __syncwarp();
  const int row = lane & 7;
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    const uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(scratch + i * 32 + row * 4));
#if defined(FLASHKDA_ARCH_150)
    asm volatile(
        "ppu.tc02.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
        : "=r"(b[i]) : "r"(addr));
#else
    (void)addr;
    b[i] = 0xdeadbeefu;
#endif
  }
  __syncwarp();
}

__device__ __forceinline__ void check_scratch_guards(
    uint32_t* scratch, uint32_t seed, int lane, uint32_t* errors) {
  if (scratch[lane - kGuard] != scratch_canary(seed, lane, false))
    atomicAdd(errors, 1u);
  if (scratch[kScratchWords + lane] != scratch_canary(seed, lane, true))
    atomicAdd(errors, 1u);
}

template <Mode InputMode>
__global__ void probe(uint32_t* output, size_t route_stride,
                      uint32_t* scratch_errors) {
  // Every supported block size uses full warps; no lane exits early.
  __shared__ __align__(128) uint32_t storage[8 * kWarpStride];
  const int tid = threadIdx.x;
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const size_t global_tid = size_t(blockIdx.x) * blockDim.x + tid;
  const uint32_t seed = static_cast<uint32_t>(global_tid / kWarp);
  uint32_t* scratch = storage + warp * kWarpStride + kGuard;
  scratch[lane - kGuard] = scratch_canary(seed, lane, false);
  scratch[kScratchWords + lane] = scratch_canary(seed, lane, true);
  __syncwarp();

  uint32_t c[4];
  if constexpr (InputMode == kRoundedAcc) {
    // Preserve the production accumulator traversal and BF16 storage boundary.
    auto acc = flashkda::make_acc16_bf16_1warp(lane);
    auto f = cute::recast<float>(acc);
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i)
      f(i) = __uint_as_float(kAccFloatBits[(seed + lane * 8 + i) & 31]);
    uint32_t halves[8];
    int pos = 0;
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(f); ++m)
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(f); ++n)
        CUTE_UNROLL
        for (int v = 0; v < cute::size<0>(f); ++v)
          halves[pos++] = flashkda::BF16(f(v, m, n)).storage;
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i)
      c[i] = flashkda::pack_half2_u32(halves[2 * i], halves[2 * i + 1]);
  } else {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      const uint32_t lo = raw_half(InputMode, seed, lane * 8 + 2 * i);
      const uint32_t hi = raw_half(InputMode, seed, lane * 8 + 2 * i + 1);
      c[i] = lo | (hi << 16);
    }
  }

  uint32_t old_x1[4], new_x4[4], shfl[4];
  legacy_batched_x1(c, old_x1, lane, scratch);
  check_scratch_guards(scratch, seed, lane, scratch_errors + 0);
  flashkda::bf16_cfrag_to_bfrag_movmatrix_batched_u32(c, new_x4, lane, scratch);
  check_scratch_guards(scratch, seed, lane, scratch_errors + 1);
  flashkda::bf16_cfrag_to_bfrag_swizzle_u32(c, shfl, lane);
  check_scratch_guards(scratch, seed, lane, scratch_errors + 2);
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    const size_t at = global_tid * 4 + i;
    output[0 * route_stride + kGuard + at] = c[i];
    output[1 * route_stride + kGuard + at] = old_x1[i];
    output[2 * route_stride + kGuard + at] = new_x4[i];
    output[3 * route_stride + kGuard + at] = shfl[i];
  }
}

struct DeviceWords {
  uint32_t* p = nullptr;
  ~DeviceWords() { if (p) hggcFree(p); }
};

#define CHECK_RUNTIME(expr) do { \
  auto status_ = (expr); \
  if (status_ != hggcSuccess) { \
    std::fprintf(stderr, "runtime failure: %s -> %d at line %d\n", \
                 #expr, int(status_), __LINE__); \
    return false; \
  } \
} while (0)

uint64_t hash_word(uint64_t hash, uint32_t word) {
  for (int b = 0; b < 4; ++b) {
    hash ^= (word >> (8 * b)) & 255u;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool run_case(Mode mode, int threads) {
  const uint32_t warps = mode == kExhaustive ? 65536u
                       : mode == kTagged ? 32u : 256u;
  const size_t logical_words = size_t(warps) * kWarp * kRegs;
  const size_t route_stride = logical_words + 2 * kGuard;
  const size_t all_words = route_stride * kRoutes;
  const int blocks = int(warps / uint32_t(threads / kWarp));
  DeviceWords output, errors;
  CHECK_RUNTIME(hggcMalloc(&output.p, all_words * sizeof(uint32_t)));
  CHECK_RUNTIME(hggcMalloc(&errors.p, 3 * sizeof(uint32_t)));
  CHECK_RUNTIME(hggcMemset(output.p, 0xa5, all_words * sizeof(uint32_t)));
  CHECK_RUNTIME(hggcMemset(errors.p, 0, 3 * sizeof(uint32_t)));
  if (mode == kExhaustive)
    probe<kExhaustive><<<blocks, threads>>>(output.p, route_stride, errors.p);
  else if (mode == kTagged)
    probe<kTagged><<<blocks, threads>>>(output.p, route_stride, errors.p);
  else
    probe<kRoundedAcc><<<blocks, threads>>>(output.p, route_stride, errors.p);
  CHECK_RUNTIME(hggcDeviceSynchronize());

  std::vector<uint32_t> host(all_words);
  uint32_t scratch_errors[3] = {};
  CHECK_RUNTIME(hggcMemcpy(host.data(), output.p, all_words * sizeof(uint32_t),
                           hggcMemcpyDeviceToHost));
  CHECK_RUNTIME(hggcMemcpy(scratch_errors, errors.p, sizeof(scratch_errors),
                           hggcMemcpyDeviceToHost));
  const uint32_t* packed = host.data() + kGuard;
  const uint32_t* routes[3] = {
      host.data() + route_stride + kGuard,
      host.data() + 2 * route_stride + kGuard,
      host.data() + 3 * route_stride + kGuard};

  uint64_t input_errors = 0, mismatches[3] = {}, x1_x4 = 0, canary_errors = 0;
  uint64_t digest[4] = {UINT64_C(14695981039346656037),
                        UINT64_C(14695981039346656037),
                        UINT64_C(14695981039346656037),
                        UINT64_C(14695981039346656037)};
  int printed = 0;
  for (int route = 0; route < kRoutes; ++route) {
    for (int i = 0; i < kGuard; ++i) {
      canary_errors += host[size_t(route) * route_stride + i] != kGlobalCanary;
      canary_errors += host[size_t(route) * route_stride + kGuard
                            + logical_words + i] != kGlobalCanary;
    }
  }
  for (uint32_t seed = 0; seed < warps; ++seed) {
    for (int lane = 0; lane < kWarp; ++lane) {
      for (int reg = 0; reg < kRegs; ++reg) {
        const size_t at = (size_t(seed) * kWarp + lane) * kRegs + reg;
        if (mode != kRoundedAcc) {
          const uint32_t in = uint32_t(raw_half(mode, seed, lane * 8 + 2 * reg))
              | (uint32_t(raw_half(mode, seed, lane * 8 + 2 * reg + 1)) << 16);
          input_errors += packed[at] != in;
        }
        // Independent scalar CPU oracle: source lane, source register and
        // source half are derived from the current tc02 pure-SHFL bit mapping.
        // No device-produced converter output is used as the expected result.
        const int src_lo = ((lane & 3) << 3) + (lane >> 3);
        const int src_hi = src_lo + 4;
        const int shift = ((lane >> 2) & 1) * 16;
        const uint32_t lo = packed[(size_t(seed) * kWarp + src_lo) * kRegs + reg];
        const uint32_t hi = packed[(size_t(seed) * kWarp + src_hi) * kRegs + reg];
        const uint32_t expected = ((lo >> shift) & 0xffffu)
            | (((hi >> shift) & 0xffffu) << 16);
        digest[0] = hash_word(digest[0], expected);
        for (int route = 0; route < 3; ++route) {
          const uint32_t actual = routes[route][at];
          digest[route + 1] = hash_word(digest[route + 1], actual);
          if (actual != expected) {
            ++mismatches[route];
            if (printed++ < 12)
              std::fprintf(stderr,
                  "mismatch mode=%s NT=%d seed=%u lane=%d reg=%d route=%s "
                  "actual=%08x expected=%08x\n", mode_name(mode), threads,
                  seed, lane, reg, route == 0 ? "old_x1" : route == 1 ? "x4" : "shfl",
                  actual, expected);
          }
        }
        x1_x4 += routes[0][at] != routes[1][at];
      }
    }
  }
  const bool pass = input_errors == 0 && mismatches[0] == 0
      && mismatches[1] == 0 && mismatches[2] == 0 && x1_x4 == 0
      && canary_errors == 0 && scratch_errors[0] == 0
      && scratch_errors[1] == 0 && scratch_errors[2] == 0;
  std::printf(
      "{\"schema\":\"tc02-c2b-production-x4-probe-v1\",\"mode\":\"%s\","
      "\"threads\":%d,\"blocks\":%d,\"warps\":%u,\"words_per_route\":%zu,"
      "\"u16_patterns_each_source_position\":%u,\"input_generator_mismatches\":%llu,"
      "\"cpu_oracle_mismatches\":{\"x1\":%llu,\"x4\":%llu,\"shfl\":%llu},"
      "\"x1_x4_mismatches\":%llu,\"global_canary_errors\":%llu,"
      "\"scratch_canary_errors\":[%u,%u,%u],"
      "\"word_digest_fnv1a64\":{\"cpu\":\"%016llx\",\"x1\":\"%016llx\","
      "\"x4\":\"%016llx\",\"shfl\":\"%016llx\"},\"pass\":%s}\n",
      mode_name(mode), threads, blocks, warps, logical_words,
      mode == kExhaustive ? 65536u : 0u,
      static_cast<unsigned long long>(input_errors),
      static_cast<unsigned long long>(mismatches[0]),
      static_cast<unsigned long long>(mismatches[1]),
      static_cast<unsigned long long>(mismatches[2]),
      static_cast<unsigned long long>(x1_x4),
      static_cast<unsigned long long>(canary_errors),
      scratch_errors[0], scratch_errors[1], scratch_errors[2],
      static_cast<unsigned long long>(digest[0]),
      static_cast<unsigned long long>(digest[1]),
      static_cast<unsigned long long>(digest[2]),
      static_cast<unsigned long long>(digest[3]), pass ? "true" : "false");
  std::fflush(stdout);
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  bool smoke = false;
  std::vector<int> thread_counts = {32, 64, 128, 256};
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--smoke") == 0) smoke = true;
    else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      const char* value = argv[++i];
      if (std::strcmp(value, "all") == 0) thread_counts = {32, 64, 128, 256};
      else if (std::strcmp(value, "both") == 0) thread_counts = {128, 256};
      else if (std::strcmp(value, "32") == 0) thread_counts = {32};
      else if (std::strcmp(value, "64") == 0) thread_counts = {64};
      else if (std::strcmp(value, "128") == 0) thread_counts = {128};
      else if (std::strcmp(value, "256") == 0) thread_counts = {256};
      else {
        std::fprintf(stderr, "--threads requires 32, 64, 128, 256, all or both\n");
        return 2;
      }
    } else {
      std::fprintf(stderr, "usage: %s [--smoke] [--threads 32|64|128|256|all|both]\n", argv[0]);
      return 2;
    }
  }
  if (hggcSetDevice(0) != hggcSuccess) return 3;
  int completed = 0;
  for (int threads : thread_counts) {
    // First establish the small mapping/canary checks; stop immediately on
    // failure, before allocating/running the exhaustive sweep.
    for (Mode mode : {kTagged, kRoundedAcc, kExhaustive}) {
      if (smoke && mode == kExhaustive) continue;
      if (!run_case(mode, threads)) return 1;
      ++completed;
    }
  }
  std::printf("{\"summary\":true,\"completed\":%d,\"smoke_only\":%s,"
              "\"timed\":false,\"pass\":true}\n", completed, smoke ? "true" : "false");
  return 0;
}
