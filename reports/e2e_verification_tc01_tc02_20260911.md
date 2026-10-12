# tc01 / tc02 end-to-end verification -- cuda_free @ fbc5803

Subject: the merge commit `fbc5803` (other parent is master `51aefe7`) after the `cuda_free` branch took in six master flashqla commits (full-chain bf16 recurrent state,
swizzle-B MMA natural order). This round's changes all land in flashqla prefill, 9 files in total; `src/decode`, `pla/decode`,
`scripts/decode`, `caselist` and `setup.py` have zero changes. Each of the two card generations was rebuilt from a clean working tree and
reran the full set of cases.

## 1. Environment

| | tc02 | tc01 |
| --- | --- | --- |
| Host | 30.70.218.2 (sz01t-swu450) | 11.165.159.10 (na131t-swu116) |
| Container | hp_ppu15_gdn17_opt_20260829 | hpwu-aiu-tc01 |
| Card | ZW-M890P × 8 (ppu0015, `__HGGC_ARCH__==150`) | PPU-ZW810E × 10+ (ppu001, `__HGGC_ARCH__==100`) |
| Working tree | `/mnt/ssd/hpwu/pla_verify_fbc5803` | `/swhome/hpwu/pla_verify_fbc5803` |
| actlize | `3rdparty/actlize` <- on-machine v1.0.0 unpacked source | same |

Deployed with `git archive fbc5803 | ssh tar -x`, then the on-machine actlize v1.0.0 was copied into
`3rdparty/actlize` (the submodule gitlink does not travel with the archive). Background jobs always use `python3 -u`
and append `echo EXIT=$?`.

## 2. Build

`python3 -u scripts/build.py` (omitting `--namespace` builds all five extensions), followed by
`python3 -u scripts/build.py --mode tuned --namespace pla.decode` to rebuild k-last.

Both machines: **5/5 extensions produced, `EXIT=0`, zero compile errors**:

```
pla/decode/_C...so                pla/decode/gdn/v_last/_C...so
pla/decode/kda_mtp_sglang/_C...so pla/prefill/flashqla/_C...so
pla/prefill/flashkdapro/_C...so
```

This round flashqla adds a `STATE_BF16` / `SBF16` template dimension, and the kernel signature,
`configure_fused_kernel_smem_once` and the three launch macros all gain parameters; building on both machines shows that the
instantiation chain of `fq::StatePool<STATE_BF16>` is self-consistent under both arches.

## 3. Case results

| Case | tc02 | tc01 |
| --- | --- | --- |
| `scripts/prefill/flashqla/e2e_acc_full.py` | **45/45 ALL PASS** | **41/41 ALL PASS** |
| `tests/test_op_forge_prefill.py --caselist caselist/prefill --model Qwen --state-dtype both` | **20890 pass / 0 fail** | **20890 pass / 0 fail** |
| `scripts/prefill/flashkdapro/e2e_acc_full.py` | E2E ACCURACY OK | **94/94 checks OK** |
| `tests/test_kda_decode_mega.py` | Failed 0 | Failed 0 |
| `tests/test_kda_mtp_sglang.py` | Failed 0 | Failed 0 |
| `tests/test_instrument_graph_capture.py` | All 3 configs survive graph capture | All 3 configs survive graph capture |
| `scripts/decode/test_fused_sigmoid_gating_recurrent.py` (v-last) | identical within tolerance | identical within tolerance |
| `scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py` | Decode Correctness Pass | Decode Correctness Pass |
| `..._k_last_packed.py` | ALL PASSED | ALL PASSED |
| `..._k_last_mtp.py` | All tests passed | All tests passed |

All `EXIT=0`.

`--state-dtype both` runs each case once on the fp32 state path and once on the bf16 state path; it is the main validation item for
this round's new feature. Per-case evidence looks like:

```
packed lengths=[...] H=64 Hg=16 cp=True state=bf16
  o_cos=1.00000/o_rel=6.17e-03  h_cos=1.00000/h_rel=4.45e-03  PASS
```

The `h_rel` of the bf16 state is stable around 4e-3, consistent with the bf16 mantissa width; no case exceeds the criterion.

## 4. Targeted confirmation of the porting decisions

**bf16 recurrent state (`fused_fwd_cute_body.inc` + `flashqla_cute_traits.cuh`)**:
master's new `StatePool` / `state_to_f32` / `state_from_f32` / `load_state_f32` /
`store_state_f32` are taken as-is, with only the 5 occurrences of `__nv_bfloat16` replaced by `__ppu_bfloat16`.
The fp32 branch outside `if constexpr (STATE_BF16)` stays byte-identical to the pre-staging form
(master itself requires this in its comments), so there is no risk of the fp32 path being affected by the bf16 changes.
Evidence: on both machines all 20890 cases pass for both fp32 and bf16, and the fp32 numerics match pre-merge.

**swizzle-B MMA natural order (c2905dc)**: master changed 4 TSM atoms from
`Swap=false` + manually permuting `bfrag[0],bfrag[2],bfrag[1],bfrag[3]` at the MMA call site to
`Swap=true` + passing arguments in natural order. This is the biggest semantic risk of this round, because this branch uses actlize's
`cute::PPU0015_TSM_LD_SWZL` rather than master's cutlass3 `cute::Acompute10500_TSM_LD_SWZL`.
Reading `actlize_v1.0.0/include/cute/arch/copy_ppu0015_aiu.hpp` word by word confirms that the actlize atom
carries the same Swap semantics this change relies on: `lbo`/`sbo` are the only things `Swap` selects
(`lbo = (Swap != Trans) ? row_stride : 1`, `sbo` complementary), `Swap32` is gated by
`sizeof(Element) == 4` and is therefore always false for the `__half` / `__ppu_bfloat16` instantiations,
and the address computation in DISPATCH never reads `Swap`. The header comment also states explicitly
"only 32b use Swap32 to represent swap vregs, 16b use lbo/sbo".
`FF_DISPATCH_ALL` in `fused_fwd_cute.cu` only instantiates with `__half` / `__ppu_bfloat16`,
so no 4-byte Element reaches this atom. Evidence: after the 4 `Swap=true` changes, accuracy passes fully on both machines.

**Host signature extension (`torch_extension_cute.cc`)**: `correct_h0` gains a trailing
`bool out_bf16` and `fused_fwd` a trailing `bool final_state_bf16`; the forward declarations take master's new
parameters but keep `hggcStream_t` as the stream type; the `py::arg` defaults in pybind were checked one by one against the positional
arguments in `ops.py`.

## 5. Notes

- `scripts/build.py --mode` only affects the instantiation set of `pla.decode` (k-last). The first time I used
  `--mode default`, which made tc01's graph capture and k-last MTP report
  `No kernel instantiation for BV=16 WN=4 SN=4 HV_PER_CTA=1`. After rebuilding with
  `--mode tuned` (generated from `lut/tuned_result_*.csv`) both passed. This is a build configuration
  issue unrelated to this round's code; the correct build mode for k-last is `tuned`.
- At runtime `PYTHONPATH=<working tree>` must **not** be prepended: the source tree's `pla/` shadows the wheel with `_C`
  in site-packages. The exception is `tests/test_instrument_graph_capture.py`, which does
  `sys.path.insert(0, repo_root)` itself; to run it, the 5 `_C*.so` files from site-packages must be copied into the
  corresponding directories of the working tree.
- `tests/test_op_forge_prefill.py` is GDN-only, while the `caselist/prefill` directory contains KDA
  caselists (Kimi-K3); running the whole directory directly produces many
  `FlashQLA does not support head configuration (H=12, Hg=12)` errors. So `--model Qwen` is added
  as a filter, and KDA prefill is covered by `flashkdapro/e2e_acc_full.py`.
- Running the whole directory as in `tests/test_op_forge.py --caselist caselist/decode` does not fit the mamba2 /
  v-last cases; the two machines fail with `KeyError: 'head_dim_qk'` (tc02) and
  `v_last bigbatch has no launch for H=4, HV=32, hv_per_cta=8` (tc01) respectively. The former is a
  caselist field convention issue; the latter is the pre-existing upper bound of `KERNEL_LAUNCH` in
  `fused_sigmoid_gating_bigbatch_kernel.cuh` only covering `{4,3,2,1}`. `git diff fbc5803^1 fbc5803` proves
  this round has zero changes on the decode side, so these are not regressions; this item is excluded from the matrix, and decode is covered instead by the individual cases in the table above.
- No performance regression was run this round, only accuracy and build verification. The c2905dc gains reported by master
  (static bf16 instantiation 4906->4649 instructions, dynamic 169.5M->138.5M, `fused_fwd_cp_cute_runtime_kernel`
  device time -10.51%) were not re-measured on this branch.
