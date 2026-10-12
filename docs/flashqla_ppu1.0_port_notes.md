# FlashQLA tc01 Porting Guide

This document only records the constraints and validation methods worth keeping long term when porting
tc02/PPU1.5 CuTe kernels to tc01/PPU1.0/1.1. The implementation in `src/prefill/flashqla/kernels/` is authoritative.

## 1. Architecture and build boundaries

| Target | Device | `__HGGC_ARCH__` | TensorCell/CuTe family |
|---|---|---:|---|
| tc01 | ZW810E / PPU1.0/1.1 | 100 | ppu0010 |
| tc02 | ZW-M890P / PPU1.5 | 150 | ppu0015 |

- Device code selects the MMA, AIU, TSM and ldmatrix atoms uniformly through `flashqla_arch.cuh`;
  do not repeat architecture checks in individual kernels.
- `-gencode arch=compute_80,code=sm_80` makes the PPU compiler generate both `ppu001` and
  `ppu0015` device images. Do not inject build-arch macros in `setup.py`, and do not build
  single-architecture wheels through `NVCC_APPEND_FLAGS=-Xhggc -mcpu=...`.
- The host pass of a `.cu` cannot see `__HGGC_ARCH__`. Differences that affect grid, shared memory or scheduling policy
  must be selected at runtime per device. If the host/device constants of one kernel cannot be shared safely,
  put the implementation in a `.cuh`, generate separate launchers from two tc01/tc02 `.cu` files, and dispatch from a stable entry;
  `kkt_solve_cute*` is the reference implementation.
- After building, check the path and SHA of the `_C.so` actually loaded, and validate the same wheel on both real devices.
- actlize gates its inline asm directly on `__HGGC_ARCH__`, so no `ACOMPUTE_VERSION` is defined or needed;
  do not hard-code 10500 into kernels.

## 2. Minimal architecture abstraction

When adding or porting a kernel, only use these architecture-neutral aliases from `flashqla_arch.cuh`:

- `PpuBf16MMA` / `PpuF16MMA`
- `PpuAiuLoad`
- `PpuTsmLdSwzl`

The accumulator/input register classes and fragment lane layout of tc01 differ from tc02.
An identical MMA mnemonic does not imply an identical register layout. Prefer letting the CuTe atom handle
`partition_fragment_*` and `cute::gemm`; when a C fragment must be reused manually as a subsequent B
fragment, a tc01-specific lane remap is required, and the tc02 shuffle must not be reused.

`cute/ppu_tensor_mix.hpp` must be included before the tc01 AIU copy-traits header,
because the latter uses `is_mix_iterator` inside a `constexpr if`.

## 3. The closed tc01 AIU/TSM data path

The tc02 `.2d.tile` AIU encoding and explicit swizzle mode are not the same interface as the tc01 tensor descriptor.
Do not port by changing inline asm parameters. tc01 must use the closed combination
validated in actlize:

```text
PPU0010_AIU_LOAD
  -> AIU-native SmemLayout
  -> PPU0010_TSM_LD_SWZL
  -> PPU0010 MMA fragment
```

Implementation guidelines:

1. Represent the gmem tile with `make_mix_tensor` and `make_counting_stride<2,1,1>()`.
2. `AiuDesc::init` takes the full tile shape, the real matrix bounds and the row pitch in elements.
3. The value layout of `make_tiled_copy` describes the payload of a single AIU instruction; let
   `cute::copy` expand multiple K slices, and do not hand-write several overlapping copies.
4. A single tc01 AIU copy in the contiguous direction is limited to 128 B. 128 b16 elements therefore need two
   64-element slices; `InstNum` must be derived from the tile width and the 128 B limit.
5. One thread issues the AIU copy; the caller is responsible for correct commit/wait/barrier.
6. `.padz` zeroes the tail rows, but the descriptor's `rows` and row stride must be the real values.

TSM written by AIU is not an ordinary packed row-major buffer. All elementwise accesses,
fragment partitions and transpose reads must use the same `SmemLayout` view as the write.
Do not disguise a tiled/swizzled layout as a transposed matrix by swapping strides.

### Swizzle must be paired

- Q/K tiles later read with `TSM_LD_SWZL` must also be written by AIU with the matching swizzle enabled.
- V/A tiles that need ordinary elementwise reads should use a linear/
  AIU-native layout consistent with the reader.
- A generic shared pointer must be converted to a CTA-relative shared address before being passed to a multi-CTA swizzled copy;
  missing `__cvta_generic_to_shared` writes into another CTA's TSM slice.

## 4. Resource and synchronization constraints

- TSM capacity, bank mapping and the number of co-resident CTAs differ between tc01 and tc02. After every scratch
  or padding increase, recompute the shared-memory footprint and update the launch bounds accordingly.
- The current tc01 fused path needs one CTA/CU and splits DV into two CTA shards; do not directly
  apply the tc02 two-CTA/CU assumption.
- fp32 rows of 128 B/32 floats keep hitting the same bank. The KKT inverse uses a 33-float
  physical leading dimension; other hot tiles likewise rotate banks through small row padding.
- Visibility across blocks must use block-level publication. An inverse version that used only warp barriers
  once produced silent errors in the second half of the rows, and must not be restored just because a microbenchmark is faster.
- Buffer aliasing may only happen where the previous stage has completed with an explicit fence; mistaking identical logical shapes
  for identical physical layouts is the most common error in tc01 porting.

## 5. Algorithm-level porting principles

- Correctness comes before instruction-by-instruction copying. tc01 KKT uses an fp32 unit-lower triangular recurrence
  and keeps the Gram tile; do not force reuse of the temporary layout of the tc02 blocked inverse.
- In GVA, several value heads share the same key head. Gram should be computed once per `(chunk, key-head)`
  and reused, avoiding recomputation per value head.
- The tc01 C/B fragment mapping, state-row layout and output store must each be validated separately. An output that looks
  close does not mean the final state is correct; `o`, `A` and `ht` are all part of the regression gate.
- The CP planner must use the actual CU count of the running device and the tc01 DV shard count. Do not use a single
  fixed CU constant to represent both tc01 and tc02.

## 6. Wrong approaches not to retry

- Hand-writing tc01 AIU asm to emulate tc02 `.2d.tile`.
- Treating AIU-native/swizzled TSM as a packed row-major buffer.
- Materializing a transpose just by swapping CuTe strides.
- Reusing the tc02 accumulator-to-B-fragment shuffle on tc01.
- Raising `MinBlocks` without recomputing TSM.
- Deciding whether a short kernel regresses from a single event timing; short shapes need reversed ordering and multi-process
  aggregation.

## 7. Dual-architecture build

Build from clean sources inside the PPU SDK container. The command below builds both FlashQLA and FlashKDApro;
the single resulting wheel contains both the tc01 and tc02 device images.

```bash
PLA_BUILD_NAMESPACE=pla.prefill python -m pip wheel . \
  --no-build-isolation -w dist
```

The compile commands in the build log should contain `-gencode arch=compute_80,code=sm_80`, and after PPU driver
expansion contain both `--ppu-arch=ppu001` and `--ppu-arch=ppu0015`. Neither the log nor the binary
should contain `FLASHQLA_BUILD_ARCH_100` or `FLASHKDA_BUILD_ARCH_100`.

## 8. Validation gates

Only two public validation entries are kept:

```bash
# end-to-end accuracy over all dispatch, CP, h0 and tail cases
CUDA_VISIBLE_DEVICES=<card> \
  python scripts/prefill/flashqla/e2e_acc_full.py

# end-to-end performance, output and state checks against the in-repo Triton reference implementation
CUDA_VISIBLE_DEVICES=<card> \
  python scripts/prefill/flashqla/bench_vs_fla.py \
    --num-v-heads 32 --num-k-heads 16 --seq-lens 2048,4096,8192
```

Before submitting a tc01 port, at least the following must hold:

1. Both tc01 and tc02 binaries build successfully from clean sources and load the expected `_C.so`.
2. E2E accuracy passes fully on both sides, covering FP16/BF16, all production dispatches, h0, CP and tail chunks.
3. tc02 has no accuracy or performance regression against the pre-port baseline.
4. tc01 performance comparisons use the same card, the same inputs and reversed run orders; short shapes crossing the threshold get
   multi-process aggregation, so frequency noise is never taken as a kernel conclusion.

When a problem needs to be located, use the CP, packed, stage breakdown, reference and hardware probe tools
under `scripts/prefill/flashqla/private/`; these are not release interfaces.
