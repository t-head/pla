# PLA cuda_free branch E2E verification report: PPU1.5 and PPU1.1 on two machines

Verification date 2026-08-31. The code under test is the `cuda_free` branch (CUDA-free sources + actlize v1.0.0 + hgcc device
compilation), commit `4b2440e`. The baseline is `master` @ `79aed40` (cutlass3 PPU fork + nvcc compatibility route).

## 1. Conclusions

1. **The rework itself introduces zero performance regression.** A/B on the same machine, same container and same bench shows cuda_free and master matching point by point,
   with differences <1% and bit-identical accuracy metrics.
2. **The accuracy gates pass fully on both card generations**; none of the three extensions exceeds any criterion.
3. **An SDK performance regression was found**: container SDK `ppu202608201210`, compared with `ppu202607250839`, makes flashkdapro 2~3x slower on the same machine,
   same driver and same card, while the FLA-Triton side is unaffected.
4. **A severe flashkdapro performance problem was found on tc01**: on PPU1.1 it is 7~25x slower than FLA-Triton, whereas on PPU1.5 it is
   1.4~2.2x faster. This is not a magnitude an SDK difference can explain, and needs dedicated optimization.
5. The v-last decode kernel is about 1.7x faster on PPU1.1 than on PPU1.5, confirmed by measurements on three machines; it is a real arch path difference.

This round covers `pla.decode.gdn.v_last`, `pla.prefill.flashqla` and `pla.prefill.flashkdapro`.
`pla.decode` (k-last) and `pla.decode.kda_mtp_sglang` are not yet torch-free, do not build under hgcc, and are out of scope.

## 2. Environment

| | PPU1.5 (A) | PPU1.5 (B) | PPU1.1 |
|---|---|---|---|
| Host | 30.70.218.2 (sz01t-swu450) | 11.160.255.130 | 11.161.48.226 (ppu810e-test008) |
| Card | ZW-M890P, 850W | ZW-M890P | ZW-810E, 400W |
| Driver | 2.2.1-rc0f8b | 2.1.2-dd40b5 | 2.1.3-rc401b |
| device arch | `__HGGC_ARCH__=150` (tc02) | 150 (tc02) | 100 (tc01) |

Containers used (all build a dual-arch `-arch=ppu_10,ppu_15` fatbin, so one artifact serves both generations):

| Container | Host | Image SDK | torch / sglang |
|---|---|---|---|
| hp_ppu15_gdn17_opt_20260829 | 218.2 | ppu202608061910 | 2.11.0 / none |
| hp_ppu15_sgl_e2e_0831 | 218.2 | **ppu202608201210** | 2.11.0a0 / 0.5.14 |
| hp_ppu15_sgl_oldsdk | 218.2 | **ppu202607250839** | 2.11.0 / 0.5.13 |
| hp_ppu15_sglang0.5.12 | 255.130 | ppu202606091910 | 2.10.0 / 0.5.12 |
| hp_pla810e_prefill_opt_20260829 | 48.226 | ppu202608061910 | 2.11.0 / none |
| hp_ppu11_sgl_e2e_0831 | 48.226 | ppu202607271910 | 2.10.0 / 0.5.12 |

At runtime `PYTHONPATH=<working tree>` must be prepended, otherwise the stale `pla` package in the container's site-packages shadows the inplace-built `_C`.

## 3. Accuracy gates

| Extension | Entry | PPU1.5 (tc02) | PPU1.1 (tc01) |
|---|---|---|---|
| flashqla | `scripts/prefill/flashqla/e2e_acc_full.py` | **45/45 ALL PASS** | **41/41 ALL PASS** |
| flashkdapro | `scripts/prefill/flashkdapro/e2e_acc_full.py` | **94/94 checks OK** | **94/94 checks OK** |
| v_last | `test_fused_sigmoid_gating_recurrent.py --dtype bf16` | identical within tolerance ×2 | identical within tolerance ×2 |

The difference between 45 and 41 for flashqla is not missing coverage: the script keys on compute capability and only adds 4 H=32
CP cases on 8.9 (tc02); on tc01 H32 stays serial. The per-section results agree on both machines: GVA 9/9, h0 3/3, Partial tail 7/7,
Indexed state 3/3, CP-path 7/7 and 3/3 respectively.

flashkdapro covers the dvsplit BV16/BV32, full-DV nows, varlen nows and baseline DVT32 dispatch branches,
plus the partial tail and final_state contracts. The `err_ratio` upper bounds are `out=6.11e-03` and `state=4.99e-03`, against the criterion
`TOL=1e-2`; the numbers of the two generations nearly coincide (tc02 6.05e-03 / tc01 6.11e-03), showing that numerical behaviour stays
aligned despite going through different trait families. All `*->none` combinations are `bit-identical=True`.

## 4. A/B: did the rework introduce a regression

Both branches were compiled in the same container on 255.130 (sglang0.5.12) and ran the same `bench_vs_fla.py`
(`--warmup 10 --repeat 50 --ragged`, H=12 D=128 bf16):

| seq_lens | master `pla_ms` | cuda_free `pla_ms` | master speedup | cuda_free speedup |
|---|---|---|---|---|
| 512 | 0.118 | 0.118 | 2.71x | 2.78x |
| 1024 | 0.181 | 0.182 | 2.24x | 2.26x |
| 2048 | 0.301 | 0.304 | 1.64x | 1.62x |
| 4096 | 0.491 | 0.489 | 1.55x | 1.56x |
| 8192 | 0.773 | 0.765 | 1.61x | 1.63x |
| 4096,137,2048,63 | 0.787 | 0.786 | 1.19x | 1.19x |

`o_err` / `state_err` are bit-identical on both sides (6.483e-02 / 4.425e-03 etc.). **The CUDA-free source rework + switching from cutlass3
to actlize has no effect on performance or numerics.**

## 5. SDK performance regression

Same machine 218.2, same driver 2.2.1, same cuda_free code, only the SDK of the container image changed:

| seq_lens | SDK ppu202608201210 `pla_ms` | SDK ppu202607250839 `pla_ms` | new/old | new speedup | old speedup |
|---|---|---|---|---|---|
| 512 | 0.230 | 0.096 | 2.40x | 0.90x | 2.15x |
| 1024 | 0.415 | 0.158 | 2.63x | 0.74x | 1.92x |
| 2048 | 0.786 | 0.281 | 2.80x | 0.50x | 1.40x |
| 4096 | 1.522 | 0.466 | 3.27x | 0.45x | 1.46x |
| 8192 | 2.243 | 0.747 | 3.00x | 0.53x | 1.60x |
| ragged | 3.078 | 0.770 | 4.00x | 0.28x | 1.13x |

The `triton_ms` of the same cases is nearly identical on both sides (0.205~1.198 vs 0.208~1.197), so this is not a machine-state or
measurement-methodology issue: the PLA kernels compiled by the new SDK are slower. Under the new SDK PLA flips straight from "faster than FLA" to "slower than FLA".
We recommend reporting this to the SDK team, and using `ppu202607250839` or an older image as the performance baseline in the meantime.

## 6. Performance comparison across the two card generations

### 6.1 flashkdapro vs FLA-Triton (warmup=10 repeat=50, H=12 D=128 bf16)

| seq_lens | tc02 `pla_ms` (218.2 old SDK) | tc01 `pla_ms` (48.226) | tc01/tc02 | tc02 speedup | tc01 speedup |
|---|---|---|---|---|---|
| 512 | 0.096 | 2.512 | 26x | 2.15x | **0.13x** |
| 1024 | 0.158 | 4.957 | 31x | 1.92x | **0.09x** |
| 2048 | 0.281 | 9.844 | 35x | 1.40x | **0.06x** |
| 4096 | 0.466 | 19.611 | 42x | 1.46x | **0.04x** |
| 8192 | 0.747 | 39.321 | 53x | 1.60x | **0.04x** |
| ragged | 0.770 | 70.049 | 91x | 1.13x | **0.02x** |

Two tc01 re-runs agree point by point (2.512/2.527, 39.321/39.322), ruling out noise and CPU contention. On tc01 PLA is
7~25x slower than FLA-Triton, the gap grows with seq, and the `ragged` batch is the worst. This matches another line of evidence: running
657 cases with `generalized_prefill_replay.py --mode perf`, the tc01/tc02 median ratio for the KDA group is
stable at 17~18x over H=12~96, whereas the GDN group is only 2.25~2.65x (see 6.2).

In other words the GDN gap between generations matches the hardware tier (850W vs 400W), while KDA is clearly disproportionate; the whole KDA path on tc01
has probably landed on an inefficient implementation. We recommend profiling a single KDA case (e.g. H=24 / T=2048) with acu on PPU1.1
to determine whether dispatch picks the wrong branch, AIU/TSM is not enabled, or it falls back to a scalar implementation.

### 6.2 Full prefill cases (657 cases, caselist/prefill shard 0/40, warmup=5 repeat=20)

Both machines `pass=657 fail=0`. Per-group medians (ms); here both machines use the vllm container with SDK `ppu202608061910`:

| family | H | tc02 | tc01 | tc01/tc02 |
|---|---|---|---|---|
| gdn | 8 / 16 / 24 / 32 | 0.507 / 0.871 / 1.300 / 1.826 | 1.314 / 2.138 / 2.924 / 4.437 | 2.59 / 2.45 / 2.25 / 2.43 |
| gdn | 48 / 64 / 128 | 2.930 / 3.304 / 7.142 | 6.840 / 8.489 / 18.901 | 2.33 / 2.57 / 2.65 |
| kda | 12 / 24 / 48 / 96 | 4.519 / 6.620 / 15.264 / 23.061 | 76.783 / 111.747 / 279.274 / 414.584 | **17.0 / 16.9 / 18.3 / 18.0** |

Overall median 1.749 ms (tc02) vs 4.462 ms (tc01); cumulative over the 657 cases 2,841 ms vs 33,451 ms.

### 6.3 v_last decode (bf16, batch=129 tp=4, hk=4 hv=8 dk=dv=128)

| Machine | arch | kernel_time |
|---|---|---|
| 218.2 | tc02 | 70.45 us / 70.49 us |
| 255.130 | tc02 | 70.23 us |
| 48.226 | tc01 | 40.53 us / 42.55 us |

The two PPU1.5 machines agree (70.2~70.5 us), so PPU1.1 being about 1.70x faster is a real arch difference, not machine state.
The direction is the opposite of prefill, and also the opposite of the hardware tier. A possible cause is that the two generations take different branches: `__HGGC_ARCH__==150` turns on
`PLA_USE_AIU` and `ENABLE_FP8`, and bigbatch's `choose_hv_per_cta()` uses the runtime CU count as a heuristic input;
the different CU counts of the two generations select different `hv_per_cta`, and tc02 may have picked a path that is not actually better. Likewise we recommend comparing the
kernel timelines and AIU utilization of the two generations with acu.

## 7. FlashQLA vs FLA-Triton (explicit dual entry)

`flashqla/bench_vs_fla.py` no longer depends on the old PPU SGLang
`SGLANG_SAIL_PLA_PREFILL` switch. The script now calls explicitly:

- PLA: `pla.prefill.flashqla.chunk_gated_delta_rule_fwd`
- FLA: the official SGLang `sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule`

Both sides share the same q/k/v/g/beta, packed offsets and slot indices, and each clones the same indexed fp32
state pool. Timing is the CUDA Event median with warmup=10, repeat=50.

| seq_lens | tc02 FLA ms | tc02 PLA ms | tc02 speedup | tc01 FLA ms | tc01 PLA ms | tc01 speedup |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 0.221 | 0.124 | 1.78x | 0.386 | 0.224 | 1.73x |
| 1024 | 0.295 | 0.209 | 1.41x | 0.501 | 0.365 | 1.37x |
| 2048 | 0.464 | 0.307 | 1.51x | 0.806 | 0.683 | 1.18x |
| 4096 | 0.802 | 0.509 | 1.58x | 1.357 | 1.259 | 1.08x |
| 8192 | 1.489 | 0.862 | 1.73x | 2.463 | 2.426 | 1.02x |
| 4096,137,2048,63 | 1.176 | 0.987 | 1.19x | 1.918 | 2.396 | 0.80x |

Accuracy passes on both generations: `o_err` is 4.754e-3~4.768e-3 and `state_err` is
3.453e-3~3.555e-3, all below the 2e-2 threshold. tc02 leads in every case; on tc01 single sequences still lead by 1.02~1.73x,
but the ragged batch trails by 20%, so the tc01 grouped/CP plan needs a follow-up check.

## 8. Reproduction commands

```bash
# accuracy (on each machine)
cd /mnt/ssd/hpwu/pla_e2e_20260831
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=4 PPU_VISIBLE_DEVICES=4 \
  python -u scripts/prefill/flashkdapro/e2e_acc_full.py
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=5 PPU_VISIBLE_DEVICES=5 \
  python -u scripts/prefill/flashqla/e2e_acc_full.py
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=6 PPU_VISIBLE_DEVICES=6 \
  python -u scripts/decode/test_fused_sigmoid_gating_recurrent.py --dtype bf16

# vs FLA (needs the sglang container; both scripts call PLA/FLA explicitly and do not depend on environment switches)
cd /mnt/ssd/hpwu/pla_e2e_sgl_20260831
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=4 PPU_VISIBLE_DEVICES=4 \
  python -u scripts/prefill/flashqla/bench_vs_fla.py --warmup 10 --repeat 50 --ragged
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=5 PPU_VISIBLE_DEVICES=5 \
  python -u scripts/prefill/flashkdapro/bench_vs_fla.py --warmup 10 --repeat 50 --ragged

# full prefill cases, standalone timing
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=5 PPU_VISIBLE_DEVICES=5 \
  python -u scripts/prefill/generalized_prefill_replay.py \
    --list-dir caselist/prefill --family all --mode perf \
    --output e2e_logs/prefill_perf.jsonl --num-shards 40 --shard-index 0 \
    --warmup 5 --repeat 20

# decode performance
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=6 PPU_VISIBLE_DEVICES=6 \
  python -u scripts/decode/test_fused_sigmoid_gating_recurrent.py --benchmark --dtype bf16
```

Log locations: `/mnt/ssd/hpwu/pla_e2e_20260831/e2e_logs/` on 218.2 and 48.226 (prefixes `tc02_*` / `tc01_*`),
`/mnt/ssd/hpwu/pla_e2e_sgl_20260831/e2e_logs/`, `/mnt/ssd/hpwu/pla_oldsdk_ab/e2e_logs/` on 218.2,
and `/mnt/ssd/hpwu/pla_master_ab/e2e_logs/` and `/mnt/ssd/hpwu/pla_cuda_free_verify/e2e_logs/` on 255.130.

## 9. Follow-ups

1. flashkdapro on tc01 is 7~25x slower than FLA; needs an acu profile to locate the cause (highest priority).
2. The KDA performance regression in SDK `ppu202608201210` should be reported to the SDK team; meanwhile use `ppu202607250839` as the performance baseline.
3. On tc02 v_last is actually 1.7x slower than on tc01; confirm whether this is a `hv_per_cta` selection or AIU path issue.
4. FlashQLA on tc01 is 1.02~1.73x faster than FLA for single sequences, but the ragged case is only 0.80x; the tc01 grouped/CP plan needs checking.
