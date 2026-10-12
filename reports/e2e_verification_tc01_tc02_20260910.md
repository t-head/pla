# tc01 / tc02 end-to-end verification -- cuda_free @ f4994ff

Subject: the merge commit after the `cuda_free` branch took in three master commits (AIU on ppu001, ZW810E tuning table,
eagle-tree fence). Each of the two card generations was rebuilt from a clean working tree and reran the full set of
cases.

## 1. Environment

| | tc02 | tc01 |
| --- | --- | --- |
| Host | 30.70.218.2 (sz01t-swu450) | 11.165.159.10 (na131t-swu116) |
| Container | hp_ppu15_gdn17_opt_20260829 | hpwu-aiu-tc01 |
| Card | ZW-M890P × 8 (ppu0015, `__HGGC_ARCH__==150`) | PPU-ZW810E × 10+ (ppu001, `__HGGC_ARCH__==100`) |
| PPU SDK | ppu202608061910 | ppu202609011910 |
| torch | 2.11.0 | 2.10.0 |
| sglang | none | 0.5.14 |
| Working tree | `/mnt/ssd/hpwu/pla_verify_f4994ff` | `/swhome/hpwu/pla_verify_f4994ff` |
| actlize | `3rdparty/actlize` <- on-machine v1.0.0 unpacked source | same |

Deployed with `git archive f4994ff | ssh tar -x`, then the on-machine actlize v1.0.0 was copied into
`3rdparty/actlize` (the submodule gitlink does not travel with the archive). At runtime `PYTHONPATH=<working tree>`
is always prepended, otherwise the stale `pla` package in the container's site-packages shadows the inplace-built
`_C`. Background jobs always use `python3 -u` and append `echo EXIT=$?`.

## 2. Build

`MAX_JOBS=64/48 NVCC_THREADS=16 python3 -u setup.py build_ext --inplace`

Both machines: **5/5 extensions produced, `EXIT=0`, zero compile errors** (the two hits for `error` in the log are
setuptools deprecation messages):

```
pla/decode/_C...so                pla/decode/gdn/v_last/_C...so
pla/decode/kda_mtp_sglang/_C...so pla/prefill/flashqla/_C...so
pla/prefill/flashkdapro/_C...so
```

Each compilation unit produces both `ppu0015` and `ppu001` code (`-arch=ppu_10,ppu_15`).

tc01 building successfully is the most important build signal of this round: the arch-100 AIU branch ported into
`pla_device_helpers.cuh` carries three `static_assert`s (no swizzle, w axis / n axis must be contiguous). tc01
compiling means every `ppu_aiu_load` call site in the repository really is a dense cube on ppu001, and there is no
strided call that would be silently mis-copied.

## 3. Case results

| Case | tc02 | tc01 |
| --- | --- | --- |
| `scripts/prefill/flashqla/e2e_acc_full.py` | **45/45 ALL PASS** | **41/41 ALL PASS** |
| `scripts/prefill/flashqla/bench_vs_fla.py` | tolerance OK, 1.09–1.66x | tolerance OK, 1.01–1.68x |
| `scripts/prefill/flashkdapro/e2e_acc_full.py` | **94/94 checks OK** / E2E ACCURACY OK | **94/94 checks OK** / E2E ACCURACY OK |
| `scripts/decode/test_fused_sigmoid_gating_recurrent.py` (v-last) | identical within tolerance | identical within tolerance |
| `scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py` | Decode Correctness Pass | Decode Correctness Pass |
| `scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py` | ALL PASSED | ALL PASSED |
| `scripts/decode/..._k_last_mtp.py` × 3 rounds | 3/3 rounds All tests passed | 3/3 rounds All tests passed |
| `tests/test_kda_decode_mega.py` | Passed 48 / Failed 0 | Passed 48 / Failed 0 |
| `tests/test_kda_mtp_sglang.py` | Passed 20 / Failed 0 | Passed 20 / Failed 0 |
| `tests/test_instrument_graph_capture.py` | All 3 configs survive graph capture | All 3 configs survive graph capture |

All `EXIT=0`.

## 4. Targeted confirmation of the three porting decisions

**AIU on ppu001 (67f4d4b -> `pla_device_helpers.cuh`)**: master changed the device helper inside `src/common.cuh`,
whereas this branch had already moved the helper into `pla_device_helpers.cuh` (so device TUs never see torch), so
the change was replayed structurally instead of taking master's file directly; `src/common.cuh` stays host-only.
The ported `ppu_aiu_load` function body is byte-for-byte identical to master. Evidence: tc01 builds (including the
`static_assert`s), and tc01 passes all cases.

**ZW810E tuning table (2178eb6 -> regenerated `generated/`)**: the LUT is taken as-is, but `generated/` was rerun
with this branch's generator (master's output still uses `at::Half` / `cudaStream_t`). The resulting combo set
matches master's 22 files one by one; `tuned_dispatch.h` differs from master in only three places:
`cuda_runtime.h->hggc_runtime.h`, `cudaGetDevice->hggcGetDevice`,
`cudaDeviceProp->hggcDeviceProp`. Evidence: the device name reported by `ppu-smi` on tc01 is exactly
`PPU-ZW810E`, which matches the dispatch key in `tuned_dispatch.h`, i.e. the new table really takes effect on tc01;
`nm` confirms all 22 combo symbols are present in tc01's `pla/decode/_C...so`, with 12 symbols each for the 7 new ones.
The first `k_last_mtp` entry in the LUT is still `BV16/WN4/SN4/HV1`, so the tc01 MTP contract has not regressed.

**eagle-tree race (5318809)**: `__threadfence()` + `__syncthreads()` merged cleanly automatically.
The fix targets run-to-run nondeterminism, so the MTP cases were run 3 rounds on each machine (6 rounds total), all passing.

## 5. Notes

- `kda_decode` got exit 5 the first time it was launched via `pytest` (the file is an argparse script, not a
  pytest case); running it directly gave 48/0. This is an invocation issue unrelated to the code.
- `src/decode/v-last/v_last_common.cuh` still does `#include <torch/all.h>`. This is pre-existing state on this
  branch and was not touched this time; the v-last TUs take the torch compat path, and both build and accuracy pass.
- No performance regression was run this round (the numbers in `reports/perf_*.csv` come from master-side machines and
  configuration); only the tolerance and relative speedup of `bench_vs_fla` were run.
