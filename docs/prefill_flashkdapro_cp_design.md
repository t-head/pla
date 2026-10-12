# FlashKDApro PPU1.5 — Intracard Context Parallel (CP) Design

Scope: `pla.prefill.flashkdapro` (KDA prefill forward, bf16, DK=DV=128, CHUNK=16).
Hardware: PPU1.5 (ZW-M890P, 39 CU). Reference implementation: cuLA's intracard CP.
Status: functionally complete, accuracy all green, dispatch gates anchored to measurements (see §7/§8/§10).

---

## 1. Motivation

KDA prefill has a two-kernel structure:

- **K1 (prepare)**: token parallel; L2-normalizes q/k, applies gating, and builds the per-chunk WY/KKT workspace. Naturally saturated, not the bottleneck.
- **K2 (recurrence)**: for each `(sequence, head)` advances the recurrent state **serially** along chunks, with the state resident in registers.

K2's concurrency is only `grid = (N, H, DV/BV)`. The most common case in real K3 (Kimi-K3) inference is **small batch / low head count / long sequence**: tp8 → H=12, tp4 → H=24, B=1, seqlen up to 32K+. K2 then has only 48~96 chains, far fewer than 39 CU × resident chains per CU; the machine idles and a single serial chain becomes the only critical path. CP **trades total work for critical-path length**: it cuts one long chain into S segments run in parallel, at the cost of rerunning the recurrence about 2.3 times.

One-line criterion: **CP only wins when the serial chain, not the machine, is the bottleneck.**

---

## 2. Mathematical basis: affine recurrence and M/B duality

The chunk-level KDA recurrence is **affine** in the state:

```
S_{t+1} = A_t · S_t + P_t · x_t
A_t = diag(g_t) − P_t · kd_t
P_t = kr_t^T · INV_t · diag(beta_t)
```

So a run of consecutive chunks composes into another affine operator `(M_i, B_i)`:

```
carry_out = M_i · carry_in + B_i
```

**M/B duality (following cuLA)**: both chains can be computed by the **same mainloop**, with no new math kernel:

| Chain | Initial value | v input | Product |
|---|---|---|---|
| B chain (bias) | 0 | real v | `B_i` |
| M chain (transition) | identity I | `v := 0` | `M_i` |

Key observation: `A_t` acts on the state **from the left**, so **each column** of the carried matrix evolves independently. This is exactly the property the existing DV split (each CTA owns a disjoint column slice of the state) already exploits, so the M chain reuses the DV split and the whole register-resident mainloop unchanged.

---

## 3. Three-pass pipeline

```
K1 (prepare, unchanged)
      │
      ▼
pass 1  pre_scan     kernel2_recurrence_nows.cu, <UseSegTable=true, PreScan=true>
        grid = (n_seg, H, 2*(DV/BV))     ← grid.z doubled
        lower half of blockIdx.z = B chain (zero init + real v)
        upper half of blockIdx.z = M chain (identity init + v:=0)
        no O written; both chains land in chain_state[chain][seg]
      │
      ▼
pass 2  merge        kernel3_cp.cu   (the only serial pass; serial length = number of segments, not chunks)
        carries[first]     = initial_state[seq] (or 0)
        carries[first+i+1] = carries[first+i] @ M_i + B_i
        grid = (DV/BVM=8, N, H), 256 threads, BVM=16, double-buffered M/B tiles
      │
      ▼
pass 3  segment      kernel2_recurrence_nows.cu, <UseSegTable=true, PreScan=false>
        grid = (n_seg, H, DV/BV)
        init_state = carries[seg] (HasStateIn unconditionally true)
        produces O (all tokens) + final_state (written only by the last segment of each sequence)
```

Two template flags turn the same kernel into two of the three passes, with no mainloop duplication:

- `UseSegTable`: `blockIdx.x` indexes a **segment descriptor** (tok_begin / seg_len / ws_tile_base) rather than a sequence, and init/final state are indexed per segment. The math is identical to serial; only the number of concurrent chains is ×S.
- `PreScan`: doubles `grid.z` to run the M/B dual chains, skipping the output store.

**Why the serial length really gets shorter**: the number of serial merge steps is `T/(CH·seg_tiles)`, whereas the serial path is `T/CH`. More segments make the merge longer but still far shorter than the original chain; the segment-length floor `CP_MIN_SEG_TILES=32` (512 tokens) keeps both the merge chain and the fixed per-segment overhead negligible.

---

## 4. Transposed storage algebra (why merge is a bare NN GEMM)

K2 stores the state **transposed**: `image[c*DK + dk] = S[dk][c]`. Let `Mst[c][dk] = M[dk][c]`; then the composition in storage coordinates reads

```
Cnew^T[dv][dk] = Σ_j Cold^T[dv][j] · Mst[j][dk] + B^T[dv][dk]
```

i.e. an ordinary **NN GEMM** whose B operand is exactly the raw `Mst` image emitted by pre_scan -- **there is not a single transpose in the whole pipeline**.

The merge epilogue accumulates in fp32 and then does a **single** bf16 rounding (`cp_store_add_f32`): the carry is the one quantity reused by every later segment, so it gets the tighter epilogue.

---

## 5. Host-side ABI

### Segment descriptor `seg_tbl` (`flashkda_ppu_util.cuh`, `KDA_SEG_STRIDE = 4` int32 per segment)

| Index | Field | Meaning |
|---|---|---|
| 0 | `tok_begin` | absolute index of the segment's first token |
| 1 | `seg_len` | number of tokens in the segment |
| 2 | `ws_tile_base` | **absolute workspace tile index** of the segment's first tile (`= tile_prefix[seq] + first local tile`), guaranteeing the segment pass reads **the same** K1 workspace |
| 3 | `out_slot` | `final_state` row to write; `-1` skips the 32 KiB state store (non-final segments have no state consumer), and in pre_scan doubles as the "no consumer for this segment" marker |

### `seq_seg` (`N × 2` int32)
`{first segment index of the sequence, number of segments}`; merge uses it to walk the segment chain per sequence.

### Segmentation rules (`plan_cp`)
- Cuts only on whole CH tiles; **only the last segment of a sequence** may end on a partial tile -- exactly the invariant assumed by the segment kernel's tail handling `actual = min(CH, seg_len − t*CH)`.
- `segs = min(mult, max(1, tiles / min_seg_tiles))`, with the tile remainder spread over the first `rem` segments.
- Gives up when `n_seg <= N` (nothing was actually split).

### Auxiliary memory
```
chain_state [2, n_seg, H, 128, 128] bf16   (the two M/B chains)
carries     [n_seg, H, 128, 128]     bf16
```
Capped at `CP_MAX_AUX_BYTES = 1 GiB`; above that CP does not engage.

### K1's `force_plain`
The CP pipeline runs the **plain (non dedicated-leader)** recurrence template, which needs the raw `INV / Mqk / beta`, so when CP engages the host must pass `force_plain=true` to K1, disabling K1's affine precomputation. This is a **must-read implicit coupling**: do not forget CP when changing the K1 affine path.

### pybind exports
- `flashkda_fwd` -- the only production entry; CP is decided entirely internally, and the Python API is unchanged.
- `flashkda_cp_plan(T_total, H, cu_seqlens?)` -- host-only probe returning `{engage, n_seg, bv, max_seg_tiles, ...}`. Tests use it to assert CP really engages; sweeps use it to read the segment count.
- `flashkda_cp_stages(...)` -- debug view that runs only the first two passes (pre_scan + merge), used for per-pass timing `t_segment = t_cp − t_stages`.

---

## 6. Exact early exit of the M chain on zero state (the key to making CP affordable)

The M chain runs with `v := 0`, so `x = (0 − kd@S_t)·beta` and `delta = P@x`. **If a CTA's state slice is entirely 0, then vsub / x / delta are all 0 and the update degenerates to `S' = S·g = 0`** -- the zero slice is an **absorbing state**; all remaining chunks and the finally stored `M_i` image are exactly zero, and the chain can stop right there.

This is not an approximation: it is an algebraically exact early exit. Under any realistic gate value, KDA's decay gate kills the transition operator within a few dozen chunks, whereas the B chain (which keeps absorbing v) must run to the end. **This is where CP drops from "rerun 3 passes" to "about 2.3 passes".**

Implementation notes (`kernel2_recurrence_nows.cu` ~L951-998):

- Check once every `ZCHK = 8` chunks to amortize the two extra barriers.
- Use a sign-masked OR-reduce: `bits |= uint32_t(rstate[...].storage) & 0x7fff`. The sign bit is masked because negative underflow leaves `−0.0`, which is equally absorbing; the only possible difference is the sign of some zero in the stored image.
- `atomicOr(sZ, 1u)`, with `sZ` borrowing `smem + S::O_scr`; both barriers are CTA-wide, so borrowing the fragment scratch does not race with its in-loop use.
- Before `break`, `tid==0` must call `k2n_aiu_wait_group<0>()` to drain the in-flight prefetch of chunk t+1 -- its target buffer is exactly the staging area the epilogue reuses for the state image.

**Performance data dependence (risk item)**: the gain depends on how fast the gate decays. With fast decay (`g_shift=0`) M dies within one chunk; with slow decay (`g_shift=-4.5`) M never dies. All performance numbers must be reported with both `slow|fast` regime bounds, never fast decay only.

---

## 7. Dispatch gate (2026-08-19 full small-BH calibration)

The generic N=1/H≤12 path keeps the two-axis rule on chain count and longest sequence:

```cpp
chains = N * H * (is_varlen ? 4 : 8)          // CTA count already provided by plain K2
                                              // varlen uses BV32 (=4 slices),
                                              // fixed-shape H<=24 uses BV16 (=8 slices)
min_seq_len(chains) = chains<=24 ? 2048
                    : chains<=48 ? 4096
                    : chains<=96 ? 32768 : INT32_MAX
```

Kimi-K3's ragged small-BH (`N=2..3, H in {12,24,48}`) is an explicit exception. It depends only on host-visible
structural quantities, never looks at input values, and uses no case-id allowlist:

| H | N | engage condition | target total segments |
|---:|---:|---|---:|
| 12 | 2 | `max_seq>=6144 && max_seq/T>=60%` | long chain 6; 10 when `max_tiles<=512` |
| 12 | 3 | `max_seq>=4096` | 13 |
| 24 | 2 | `max_seq>=4096 && max_seq/T>=80%` | 9 |
| 24 | 3 | `max_seq>=4096 && max_seq/T>=65%` | 9 |
| 48 | 2 | `max_seq>=4096 && max_seq/T>=80%` | 8 |
| 48 | 3 | `max_seq>=4096 && max_seq/T>=65%` | 8 |

The segment count per sequence is not a uniform `mult`; it is allocated in proportion to length from the longest chain's target segment length.
So `[31801,967]` only splits the long chain and does not rerun the 967-token tail five times. The target total segment count is not a
shape lookup either: the larger H is, the more `H*4` BV32 chains each segment already produces, so the calibrated points drop from
13 at H12 to 9/8 at H24/H48.

After AUTO engages, the following must still hold: `mult>=3`, auxiliary memory `<=1 GiB`, `n_seg>N`. Non-target shapes
keep the original gate; `plan_cp_dispatch` first does a cheap pre-filter on `(T,H,N)`, and only ragged
small-BH candidates that may hit read `cu_seqlens` to the host to compute dominance.

Why the 3rd condition is "longest sequence" rather than "total tokens": CP shortens the chain of **one** sequence. A batch of many short sequences is exactly the case it cannot help (N=2/H=6/2048 per sequence measured 0.91–1.05×; the same 48 chains with 8192 per sequence reach 1.18–1.49×).

The generic path still has one fixed-shape exception:

- The slow crossover for fixed N=1/H=12 has moved down from 32K to **16K**, so that shape
  uses a 16K floor; 8K slow still regresses about 2.9% and stays off.

The old conclusion "varlen H12/N>=2 and H24 are always off" came from equal-length synthetic/slow-decay sweeps and cannot
represent Kimi-K3's dominant ragged tail. ACU shows the real losing cases are a tail-latency problem at 0.308 wave/CU with the longest chain taking
97%; proportional CP clearly wins in this window, so the new calibration above overrides the old conclusion.

### Constants and calibration

| Constant | Value | Source |
|---|---|---|
| `CP_CTAS_PER_CU` | 10 | acu-measured K2 (H=12/T=32K/B=1: 96 chains → 3.777 ms serial, vs a parallel-fill floor of ~0.91 ms for the same total work, i.e. ~4× headroom, reached at roughly 10 resident chains per CU) |
| `CP_KIMI_SEG_MULT` | **13** | `13*H*4` on 39 CU is exactly 8/16 full waves for H6/H12; beats the old 17/9 across fixed/varlen, N1/N2 and slow/fast sweeps |
| `CP_MIN_SEG_TILES` | 32 (512 tokens) | segment-length floor, keeping the merge chain and fixed per-segment overhead negligible |
| `CP_MIN_SEG_MULT` | 3 | break-even of 2.3× rerun + merge |
| `CP_MAX_H` | 12 | generic path cap; the H24/H48 small-BH in the table are structured exceptions |
| `CP_CHAINS_MAX` | 96 | generic path cap; the ragged small-BH exception must still pass the dominance gate |
| CP `bv` | **32, unconditionally** | see below |

Calibration quality: the final segment targets come from exact-shape A/B followed by cross-validation over the full affected set; the acceptance condition
is `FLA/PPU>1.20` for every row under deterministic valid inputs, with boundary subsets re-verified with 20 samples.

### CP's BV must be 32 (a pitfall we hit)

CP's `bv` was once "mirroring plain K2's `auto_bv`", and plain picks BV16 for fixed-shape H≤24. Measurements show BV16 is the **wrong tile** for CP: on the fixed-shape sweep it dragged AUTO down to 0.77–0.86× of serial, while the same shapes reach 1.11–1.25× under BV32, and BV32 is the oracle's choice at **every point of every sweep**. Reason: CP can hide only 1/3 of the chains per pass, so it needs a wider tile, not a finer split.

### Environment variables (for sweeps, no rebuild needed)

| Variable | Default | Description |
|---|---|---|
| `PLA_FLASHKDA_CP` | 1 | 0=off, 1=AUTO, 2=force (bypasses the H cap and allowlist, keeping only the structural `mult>=2`) |
| `PLA_FLASHKDA_CP_BV` | 32 | allows 16/32/128 |
| `PLA_FLASHKDA_CP_MULT` | auto | forced segment multiplier |
| `PLA_FLASHKDA_CP_SEG_TILES` | 32 | segment-length floor (tiles) |
| `PLA_FLASHKDA_CP_MAX_H` | 12 | H cap |
| `PLA_FLASHKDA_CP_VERBOSE` | 0 | prints one `T/H/N/n_seg/bv` line when engaged |

`plan_cp_dispatch` also has a cheap pre-filter on **purely host-known quantities**, so non-target shapes do not even pay the D2H read of varlen `cu_seqlens`.

---

## 8. Exact-shape A/B calibration

Inputs match the final producer: q/k=1, v=0.25, g/beta/A_log/dt_bias=0, initialized outside the timed region;
`speedup=FLA-Triton/PPU`. Representative with-state results:

| H,N,seq_lens | serial | final CP candidate | FLA/CP |
|---|---:|---:|---:|
| 12,3,[8744,7208,432] | 1.592 ms | 1.549 ms (neighbor point at 12 segments) | 1.292× |
| 12,3,[17490,14417,861] | 3.118 ms | 2.933 ms (neighbor point at 12 segments) | 1.305× |
| 24,2,[15900,484] | 3.258 ms | 2.924 ms (neighbor point at 10 segments) | 1.303× |
| 24,3,[3976,26215,2577] | 6.727 ms | 5.647 ms (8 segments) | 1.354× |
| 48,2,[15900,484] | 6.797 ms | 5.608 ms (8 segments) | 1.413× |
| 48,3,[1988,13107,1289] | 7.051 ms | 5.392 ms (8 segments) | 1.438× |

BV16 regresses consistently on CP; BV32 stays fixed in the end. Non-CP H12 ragged N2/N3 uses BV16 +
DKSplit4 (all warps compute, no dedicated leader) to cover the balanced batches outside the dominance gate.

---

## 9. Full re-verification of the AUTO gate

Kimi-K3 has 1,296 unique shapes per tp, each run as two rows, no-state and with-state. The final
full set affected by the gate and closest to the threshold uses 20 samples:

| Subset | Rows | `<1.20` | Worst FLA/PPU |
|---|---:|---:|---:|
| tp8 / H12 / N2 | 304 | 0 | 1.2217× |
| tp8 / H12 / N3 | 402 | 0 | 1.2123× |
| tp4 / H24 / N2 | 304 | 0 | 1.2367× |
| tp4 / H24 / N3 | 402 | 0 | 1.2824× |
| tp2 / H48 / N3 | 402 | 0 | 1.2301× |

The final all-tp aggregated data is in `profile/small-bh-baseline-20260819/results/` and `handoff.md`.
The old benchmark's `torch.empty` inputs and the fp32→bf16 state conversion inside the timed region produced 0.53–0.99×
fake losing cases, which were removed from the acceptance criteria; the corresponding three N6/N7 outliers were retested with 50 samples on a single card at
1.557×/1.664×/1.524×.

---

## 10. Accuracy validation results (all green, TOL = 1e-2)

| Test | Result |
|---|---|
| `test_cp_correctness.py` | 28/28 (ref 8/8, carry 4/4, long 6/6, varlen 5/5, ragged 3/3, bv 2/2). E.g.: `T=16384 H=6 state=1 slow → out=7.252e-03 state=7.083e-03 Mmax=2.222e-04` |
| `test_fwd_cases.py` | fingerprints 6.060 / 6.041 / 6.566 / 6.156 / 6.064 e-3, ALL OK (stable per shape, identical to before the refactor) |
| `test_dispatch_paths.py` | 8/8 OK |
| `e2e_acc_full.py` | 20/20 OK |

The CP correctness test checks twice: (1) CP vs the serial recurrence (serial is already aligned with torch_ref; this is the tight check, comparing both `out` and `final_state`); (2) CP vs the torch reference (small cases, catching systematic errors shared by both paths).

**Tests must tune the gate (vacuity guard)**: with the default `g = randn` the per-token gate is about -3.6 and `exp2(16 token cumsum) ~ 1e-17`, so the state dies completely within one chunk -- every cross-segment carry is then numerically zero, CP is bit-identical to serial no matter what merge computes, and **the test proves nothing**. So `g_shift` pushes `g + dt_bias` negative, driving the gate toward 1 and keeping the state alive, and the test runs across a decay sweep; the slow-decay end is what really exercises carry composition. The test also first queries `flashkda_cp_plan` to assert CP really engages (with `PLA_FLASHKDA_CP=2` + a small segment-length floor).

---

## 11. Paths rejected by measurement / removed

Per the strict code-hygiene rule, optimizations rejected by measurement **leave no gated dead code**; only the conclusion stays in comments:

- **BV16 for the CP recurrence**: see §7; measured to drag AUTO down, replaced by unconditional BV32. Note this and
  the final merge's `BVM=16` are two different tile dimensions.
- **CP-only shuffle fragment swizzle**: although it removed movmatrix's TSM scratch,
  H12/32K varlen slow/fast regressed 14.0%/13.6% respectively; the implementation has been removed entirely.
- **M-chain `ZCHK=4/16`**: 4 vs 8 all falls within ±0.25% noise; 16 regresses slow by
  about 0.6%--0.7%; both removed, keeping `ZCHK=8`.
- **fixed H12 8K AUTO floor**: fast could win about 15%, but slow regresses about 2.9%, so it cannot be enabled.
- **`skip_qd` in `k2n_load_chunk`** (pre_scan does not need qd, intended to save one load): no measured gain; the implementation was removed, leaving only a negative-result comment in the source.
- **Sharing/reusing K1 with CP** (probed by `cp_k1_share.py`): K1 is not the bottleneck and CP must use `force_plain`, so sharing has no recoverable time.
- **"Pure concurrency argument" style gating** (comparing occupancy only): loses in the balanced H24/N≥2 region; finally replaced by
  structured rules on the longest-chain ratio, length floor and H-dependent segment target.
- **The 1.8–2.2× "gain" at H=8/16**: an illusion, coming from an unoptimized serial baseline (see the §8 pitfall).

---

## 12. Known limitations and future optimization directions

1. The generic gate still only covers **H≤12 / chains≤96**; Kimi-K3's H24/H48 only open the
   dominance-calibrated N2/N3 windows. N≥4 or other H are not extrapolated and default to the already faster plain path.
2. **`ZCHK=8` has completed the 4/16 sweep and is kept**; the remaining early-exit headroom is limited by when the exact zero state appears.
3. **merge is still a purely serial GEMM chain**, but has been tuned to `grid=(8,N,H)`, BVM16, 256
   threads; ACU duration at the 13-segment H12 target point is 78.4 us. Going further with a Blelloch scan
   would add intermediate states and launches, and the current absolute ceiling is very low.
4. **The B chain has no early exit** (mathematically impossible: it keeps absorbing v). The real saving would be sharing one pass between the B chain and the segment pass.
5. **Auxiliary memory 3×n_seg×H×128×128 bf16**: H=12/n_seg=64 is already 384 MiB. There is no chunking strategy beyond the 1 GiB cap.
6. **`no_state` and `with_state` take the same path** (the segment pass's `HasStateIn` is unconditionally true, and the first carry element is 0 when `initial_state` is absent); their measured performance is within noise.

---

## 13. File index

| File | Role |
|---|---|
| `src/prefill/flashkdapro/host/torch_extension_cute.cc` | CP planning (`plan_cp` / `plan_cp_dispatch` / `CpPlan`), three-pass orchestration, `flashkda_cp_plan` / `flashkda_cp_stages` probes, comments with the measured crossover table |
| `src/prefill/flashkdapro/kernels/kernel2_recurrence_nows.cu` | register-resident mainloop + the two CP template modes `UseSegTable` / `PreScan` + M-chain early exit + `launch_flashkda_cp_prescan` / `launch_flashkda_cp_segment` |
| `src/prefill/flashkdapro/kernels/kernel3_cp.cu` | serial carry merge (`BVM=16`, 256 threads, double buffering, fp32-accumulating epilogue) |
| `src/prefill/flashkdapro/kernels/flashkda_ppu_util.cuh` | `KDA_SEG_*` segment descriptor ABI |
| `src/prefill/flashkdapro/kernels/kernel1_prepare.cu` | `force_plain` parameter (disables affine precomputation when CP engages) |
| `scripts/.../private/cp_crossover_sweep.py` | CP-vs-serial crossover oracle sweep (source of the gate data) |
| `scripts/.../private/cp_stage_timing.py` | per-pass wall-clock breakdown (`t_segment = t_cp − t_stages`) |
| `scripts/.../private/test_cp_correctness.py` | CP accuracy regression (with decay sweep and vacuity guard) |

---

## 14. TP1/TP2 large-batch, long-sequence plain-path calibration (2026-08-21)

This bucket does not enable intracard CP; it optimizes K1 prepare and K2 serial
recurrence outside CP:

```text
is_varlen && B >= 16 && T_total >= 16384 && H in {48, 96}
```

### K1: 128-thread prepare

The old K1 used 256 threads. Baseline ACU on TP2/H48/B218/T32768 shows theoretical/achieved
occupancy of 100%/97.24%, but a `sync` stall ratio of 2.66 and DRAM read at only 9.56% of peak,
meaning the four extra warps mainly add CTA-wide barrier and workspace store waits rather than hiding
HBM latency. The target bucket now defaults to 128 threads; other shapes keep the original gate and 256-thread default.

After the optimization, K1 on the same shape drops from 4.179 ms to 3.556 ms, the `sync` stall ratio drops to 0.44,
and CU throughput rises from 81.6% to 85.7%. In an exhaustive 64/96/128/256 hardware sweep, 128 is consistently best.

### K2: conditional BV32

Full-DV K2 uses 128 threads and 248 registers/thread, with theoretical/achieved occupancy of only
25%/24.83%. BV32 splits each `(seq,head)` into four independent DV slices using 64-thread CTAs,
bringing registers/thread down to 184; TP2/B218/state K2 drops from 4.627 ms to 4.244 ms.

Full-bucket re-verification found the split gain depends on initial state and batch:

```cpp
is_varlen && B >= 16 && T_total >= 16384 && (H == 48 || H == 96)
    && (has_state_in || B <= 40)
```

For `no_state && B>40` the original `(B,H)` grid is already large enough, and BV32 only rereads the shared workspace;
it measured a 0.4%--1.1% regression at B=109/110, so it is explicitly excluded. The state-in path improves at least 1.5%
on every tested B. The environment variable `PLA_FLASHKDA_K2_DVSPLIT=0/16/32` still allows rollback and A/B.

### Final effect and stopping condition

Paired results against the old dispatch with the same final binary, 20 samples, 256 rows each for TP1/TP2:

| TP | Worst gain | Median gain | Geomean gain | Regressions | CV≥2% |
|---:|---:|---:|---:|---:|---:|
| 1 / H96 | 7.81% | 14.25% | 13.02% | 0 | 0 |
| 2 / H48 | 7.54% | 14.37% | 13.05% | 0 | 0 |

The remaining K2 cost is the four DV slices rereading the shared workspace: after the optimization ACU DRAM read grows from
2.20 GB to 6.56 GB, with read utilization reaching 47.2%. BV64/64, BV64/128, BV16 and
DKSplit2/4 are all slower than BV32, regressing 2.5%--4.0%, 18%--21% and 12%--48% respectively. So
BV32 sits at the discrete optimum of the current implementation, and this round keeps none of the rejected BV64 templates.

---

## 15. TP1/TP2/TP4/TP8 short-sequence, small-workload calibration (2026-08-21)

The target bucket is every `T_total<=4096` shape in the Kimi-K3 production list. Each TP has
90 fixed + 55 ragged, 145 shapes and 290 state-mode rows in total. Baseline ACU on
TP8/H12/B1/T242 shows K2 shared-state has only 48 CTAs (0.308 wave/CU) while using
512 threads and 58.1 KiB TSM; sync stall 2.658, DRAM read at only 1.66% of peak. So this bucket
is launch/recurrence/sync/TSM latency bound, not HBM bound.

### K1 prepare thread count

K1 128/256 A/B over the full varlen bucket improved 110/110 on all four TPs, so
`is_varlen && T_total<=4096` defaults to 128 threads. Low-head-count fixed tiny shapes cannot uniformly
drop warps; the continuous crossovers found by a 64/96/128/256 hardware sweep are:

```cpp
H == 96 && T >= 128
H == 48 && T >= 192
H == 24 && T >= 448
H == 12 && T >= 512
```

Below the threshold stays at 256 threads; there is no case-id special casing.

### fixed K2: nows, BV and DK workers

The old `T<256` path for H12/H24 was the 512-thread shared-state kernel; it now uses nows.
After BV16/BV32 × DK1/2/4/8, both state modes and 40--60
samples on the exact production tiny lengths, the final discrete intervals are:

| H | work interval | AUTO |
|---:|---|---|
| 96 | state `T>=48` or any mode `T>=160`, and `T<=4096` | BV32/DK1 |
| 48 | `T<256` (`T>=256` continues the existing gate) | BV32/DK1 |
| 24 | `T<32` | BV16/DK4 |
| 24 | `32<=T<64` | BV32/DK1 |
| 24 | `64<=T<128` | BV32/DK2 |
| 24 | `128<=T<=4096` | BV16/DK4 |
| 12 | no-state `32<=T<64` | BV32/DK1 |
| 12 | other `T<4096` | BV16/DK4 |

### high-H short varlen

Forcing BV32 on the 55 ragged production shapes of TP1/H96 and TP2/H48, then tightening with same-process
50-sample pairing down to a zero-regression structural region:

```cpp
is_varlen && T_total <= 4096 && (
    (H == 96 && has_state_in && N >= 3) ||
    (H == 48 && (has_state_in || N <= 3)))
```

H96/N2 and other H48 no-state batches regress due to shared workspace rereads and keep full-DV.
The new region uses BV32/DK1 uniformly; BV16 and DK2/4/8 were all swept. The last H48/B2/DK2 candidate regressed on 27
of 52 rows, worst 0.951×, so it was not rolled out.

### Final effect and plateau

Same final binary, 30 samples, counting only rows that actually switched templates:

| TP/H | Rows | Worst old/final | Median | Geomean | Max | Regressions |
|---:|---:|---:|---:|---:|---:|---:|
| 1 / 96 | 284 | 1.0258× | 1.1226× | 1.1184× | 1.2127× | 0 |
| 2 / 48 | 290 | 1.0260× | 1.0810× | 1.1026× | 1.4121× | 0 |
| 4 / 24 | 290 | 1.0195× | 1.0716× | 1.0885× | 1.5473× | 0 |
| 8 / 12 | 290 | 1.0127× | 1.1336× | 1.1187× | 1.6268× | 0 |

Representative K2 ACU drops from 63.694 us to 31.495 us (2.022×), grid 48→96, block
512→128, TSM 58.1→38.7 KiB, sync 2.658→1.149, TSM dependency
0.525→0.298. All adjacent compiled templates for K1 threads, fixed BV/DK and varlen BV/DK have been
exhausted; adding more DK workers or slices cannot keep the whole bucket monotonic, so this loop stops here. The full experiment ledger
is in `docs/short_workload_plan.md`.
