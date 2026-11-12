# GDN prefill accuracy with near-collinear keys

Date: 2026-11-12.

PLA GDN prefill exceeds the 1% group-relative-RMS gate on six frozen
near-collinear/high-beta synthetic inputs on both PPU-ZW810E and ZW-M890P.
Two ordinary controls pass on each device. A separate first-layer experiment
using reconstructed activations from fixed public model weights also finds
final-state groups above 1% for Qwen3.8-Max. These are numerical accuracy
findings; no complete-model perplexity or task-accuracy change was measured.

This PR adds documentation and a diagnostic reproducer. It does not change or
fix a kernel. The historical measurements below are pinned to PLA commit
`6658b13a693cadfe0588cd3797f10a6e0bdea0c0`, not to a newly built current-master
binary. The PR base is `5b35575c827a3033dfb5c364af6a39f9d5db107e`; its GDN changes
since the measured commit add admission and instantiation for the unrelated
`(Hv,Hk)=(16,16)` geometry. The measured geometries are `(6,2)` and `(16,2)`.

## Mathematical and comparison contract

For a value head, use physical state `S` with shape `[V,K]`, activated
`beta`, and natural-log forgetting gate `g`. The reference recurrence is:

```text
alpha_t = exp(g_t)
D_t     = alpha_t * S_(t-1)
S_t     = D_t + beta_t * (v_t - D_t @ k_t) outer k_t
o_t     = scale * S_t @ q_t
scale   = 1 / sqrt(K)
```

Each key/query head serves a contiguous group of value heads:
`key_head = value_head // (Hv / Hk)`. The tests use `K=V=128`, BF16 Q/K/V,
FP32 activated beta/g, and FP32 initial/final states. Q/K in the fixtures are
already normalized; the PLA call uses `normalize_qk=False`. Scaling is applied
once. The adapter transposes the reference `[B,Hv,V,K]` state to the public
legacy `[B,Hv,K,V]` layout before the call and transposes the result back.

The 1% gate is evaluated independently for every sequence/value-head group:

```text
relative_RMS = RMS(actual - reference) / max(RMS(reference), 1e-8)
```

For O, RMS reduces over the sequence's `token x V` elements. For final state,
it reduces over `V x K`. The maximum group error is compared with `0.01`.
This is neither elementwise relative error nor a fraction of incorrect tokens.
The historical synthetic comparator reduces FP32 tensors; references were
exported as FP32 after the independent reference calculation. Empty-sequence
state preservation is checked separately. The stress-state ranges below omit
empty sequences so that a distinct initialization defect does not dominate
the near-collinear result.

## How the investigation started

Six synthetic joint stress inputs initially exposed accuracy sensitivity in
an H200-oriented GDN experiment. That observation motivated a same-input PLA
check on both PPU models. It was not a proof of a single root cause, a failure
rate for real traffic, or a conclusion that near-collinearity alone makes the
mathematical recurrence unstable. This PR does not provide a reproducer for
that separate H200 native implementation.

The six inputs combine two TP8 head geometries with three length patterns:

| Geometry | Hk=Hq | Hv | Sequence lengths |
|---|---:|---:|---|
| Qwen3.8-Flash-Next TP8 | 2 | 6 | `[128]`, `[8192]`, `[1,65,127,0]` |
| Qwen3.8-Max TP8 | 2 | 16 | `[128]`, `[8192]`, `[1,65,127,0]` |

The model labels specify geometry here; these inputs are synthetic. The joint
stress constructor uses seed 101, nonzero initial state, and:

```text
k_t  = BF16(normalize(base_key_head + 0.003 * noise_token))
beta = 0.85 + 0.15 * sigmoid(randn)
g    = 0.0625 * logsigmoid(randn)
```

Two ordinary-key/gate `[128]` inputs, one per geometry, are controls. The
historical tensors were generated once with CUDA RNG in the original Torch
environment, then frozen with independent references and SHA256 checksums.
Both PPUs received identical fixture bytes. Short/ragged references use FP64
token recurrence; the 8192-token references use an independent FP32 64-token
triangular solve. Short cases therefore establish the finding independently
of the long-case FP32 reference.

## Historical PPU result: identical source, binary, and inputs

| Item | Recorded identity |
|---|---|
| PLA source | `6658b13a693cadfe0588cd3797f10a6e0bdea0c0` |
| ACTLIZE gitlink | `334d6980a31b40cb97b86480768297bdeacbcba8` |
| Loaded FlashQLA binary, both PPUs | SHA256 `d17fa34f2b00f5b557327842904aa86856d5bb5071a9593a55b90d9d4b0c73fe` |
| Build namespace | `pla.prefill.flashqla`, dual `ppu_10` / `ppu_15` targets |
| PPU Torch / SDK compiler | Torch 2.13.0 / SDK compiler `2.2.0-663c22`, runtime API v3 |
| PPU-ZW810E | reported CC 8.0, 64 multiprocessors |
| ZW-M890P | reported CC 8.9, 39 multiprocessors |

The same binary was verified and loaded on both PPUs. All 18 recorded
FlashQLA source hashes matched. The build needed a host-header compatibility
shim for that Torch/SDK image: the SDK CUDA include directory and preincluded
`acsolverDn.h`. Device compiler flags and numerical sources were unchanged.
These runs measure accuracy, not isolated performance.

Each device ran eight distinct inputs with `auto_cp=False` and `auto_cp=True`:
12 stress calls and four control calls. A public request for `auto_cp=True`
does not imply that the planner selected chunk parallelism (CP).

| Device | Distinct stress inputs failing | Distinct controls passing | Stress O maximum-group RMS range | Nonempty state maximum-group RMS range |
|---|---:|---:|---:|---:|
| PPU-ZW810E | 6/6 | 2/2 | 1.06592-1.14326% | 1.43087-1.72262% |
| ZW-M890P | 6/6 | 2/2 | 1.04065-1.17004% | 1.35836-1.65595% |

Those ranges cover both public settings. Default public dispatch produced:

| Geometry / lengths | 810E actual CP | 810E O / nonempty state | 890P actual CP | 890P O / nonempty state |
|---|---|---:|---|---:|
| Next / `[128]` | No | 1.06592% / 1.51792% | No | 1.04065% / 1.46781% |
| Next / `[8192]` | No | 1.09047% / 1.43087% | Yes | 1.09212% / 1.35836% |
| Next / `[1,65,127,0]` | No | 1.14326% / 1.55631% | No | 1.08362% / 1.58973% |
| Max / `[128]` | No | 1.08254% / 1.47291% | No | 1.12191% / 1.58402% |
| Max / `[8192]` | No | 1.08618% / 1.72262% | Yes | 1.07595% / 1.63523% |
| Max / `[1,65,127,0]` | No | 1.13257% / 1.47472% | No | 1.17004% / 1.44069% |

All 810E calls selected non-CP. This does not qualify a forced 810E CP
candidate. On 890P, only the two default 8192-token calls selected CP; forced
non-CP also failed. Both settings passed the controls, whose O/state errors
were below 0.40%. All reported nonempty numerical results were finite.

Compact historical evidence is in
[historical_stress_summary.json](gdn_collinear_accuracy/historical_stress_summary.json).

### Separate empty-sequence defect

On both devices, the two historical ragged `auto_cp=False` calls also returned
an incorrect final state for the empty sequence, with approximately 100%
relative RMS. The subsequent default calls selected the same non-CP route
but happened to preserve that state. The source allocates the legacy state
output with `torch::empty`, while the empty-sequence kernel branch returns
before loading h0 or storing final state. Allocator reuse is consistent with
the apparent later success; CP did not repair the initialization gap.

The reproducer reports this contract separately. An empty sequence must
preserve its initial state regardless of K similarity, and an allocator-
dependent pass is not evidence of a fix. The PR leaves this issue unchanged.

## Do model-weight activations reach a related regime?

A separate experiment reconstructed the first linear-attention layer from
fixed public BF16 weights for Qwen3.8-Flash-Next and Qwen3.8-Max. Each model
used the same 12 fixed English, Chinese, and code texts: 5110 tokens total,
226-512 tokens per independent sequence, with no chat template. This is a
small convenience corpus, not a random production sample or an online
activation dump. No full checkpoint or complete-model inference was used.

The primary boundary uses FP32 CPU projections rounded to BF16, fused-conv
FP32 SiLU then BF16, FP32 sigmoid beta, and FP32 L2 normalization followed by
BF16 Q/K. These choices differ from the literal HF fallback's rounding order
and do not reproduce deployed GEMMs bit for bit. Reconstructed K/beta/g
matched the earlier frequency capture bit for bit.

Similarity was measured on full, sequence-local 64-token chunks, excluding
tails. Stored keys were normalized in FP64, and signed cosine was averaged
over all 2016 distinct token pairs per key head. There are 78 full chunks and
1248 chunk/key-head units per model:

| Model | Mean pair cosine >=0.99 | Mean >=0.999 | Minimum pair cosine >=0.99 | Heads containing mean >=0.99 events |
|---|---:|---:|---:|---|
| Flash-Next | 91/1248 | 0/1248 | 65/1248 | K8: 78; K12: 13 |
| Max | 78/1248 | 0/1248 | 3/1248 | K2: 78 |

Head IDs are zero based. If a joint candidate additionally requires
`min(beta) >= 0.85` throughout the 64-token block for a value head, the counts
are Next 273/3744 and Max 7/9984 chunk/value-head units. These descriptors
are not calibrated predictors of kernel failure. Mean cosine >=0.99 does
not establish equivalence to all conditions in the six synthetic stress
inputs, which also fix gate strength, forgetting, initial state, and geometry.

## Measured first-layer propagation, not complete-model quality

Real-weight inputs used independent sequences with h0=0. Each TP8 head slice
was passed through the actual PLA kernel and then reassembled; all calls
selected non-CP. The reference was an independent FP64 triangular solve,
cross-checked by FP64 token recurrence. Audited O/H discrepancies between
the two references were below 1e-15.

The shared output tail starts from `BF16(oracle O)` and applies gated
RMSNorm, out_proj, and attention residual/HC merge, preserving the selected
HF BF16 boundaries. The same tail arithmetic was used for both PPU outputs
on a separate CUDA analysis device with TF32 disabled. This is not a PPU
deployment test of the whole tail, and does not reproduce TP all-reduce
ordering. It stops before the layer's MLP: "attention merge" below does not
mean a complete decoder layer.

All entries below are percentages. O/H maxima are per complete
sequence/value-head. Merge global RMS reduces over all elements; merge
P99/max reduce over each token's complete merged activation vector.

| Model | PLA device | O worst seq/head | H worst seq/head | Attention merge global RMS | Token P99 / max | Final-state groups >1% |
|---|---|---:|---:|---:|---:|---:|
| Next | 810E | 0.3875% | 0.7762% | 0.4511% | 0.5545% / 0.6861% | 0/576 |
| Next | 890P | 0.3740% | 0.4610% | 0.4514% | 0.5516% / 0.6727% | 0/576 |
| Max | 810E | 0.4142% | 1.1884% | 0.2527% | 0.7436% / 1.4426% | 1/1536 |
| Max | 890P | 0.4142% | 2.4798% | 0.2517% | 0.7437% / 1.4426% | 4/1536 |

The Max state failures were checked against FP64 token recurrence; they are
not a transcription error. Most affected value heads map to K2. Max merge
error exceeded 1% at 12/5110 tokens on 810E and 13/5110 on 890P. Those are
sample counts, not online occurrence rates. Compact evidence is in
[model_impact_summary.json](gdn_collinear_accuracy/model_impact_summary.json).

RMSNorm largely cancels a positive, purely proportional error when its
epsilon is negligible, but not a directional error. The output gamma is a
direct learned weight, with SiLU gating for Max and sigmoid gating for
Next. Neither a fixed elementwise gate nor a residual connection guarantees
that relative RMS decreases. For Next, the HC branch additionally multiplies
the attention output by fixed per-token/stream injection weights before the
residual add. The full merge metrics include all kernel error on these
inputs; they cannot all be attributed to near-collinear keys.

A hybrid-output diagnostic retained measured O only in the selected
64-token/head regions and restored BF16 oracle O elsewhere. For the strict
joint mask above, merged global RMS was about 0.228% for Next and 0.01205%
for Max on either PPU. With only the cosine mask, Max was 0.04692-0.04795%.
These are visible perturbations in selected regions, not causal estimates
of a particular algorithm's near-collinear error. Projection, normalization,
and BF16 rounding make such contributions non-additive.

A fixed-input carry diagnostic also propagated actual-minus-reference
midpoint states with FP64 recurrence and no further injected kernel error.
The resulting later-output global RMS was 0.02162-0.03231% on 810E and
0.02209-0.03446% on 890P. Max K2 could retain 92.16% / 97.50% of the initial
state-error norm after 32 further tokens, while the selected Next heads
decayed much faster. This is not an actual decode A/B experiment.

For identical later inputs, the state difference obeys:

```text
deltaS_t = exp(g_t) * deltaS_(t-1) * (I - beta_t * k_t * k_t^T)
```

If `0 <= beta_t * ||k_t||^2 <= 2`, its Frobenius norm is bounded by the
previous norm times `exp(g_t)`. Near-collinearity does not, by itself, imply
exponential growth of an existing exact-recurrence error. Real execution can
continually inject new numerical error, and perturbed later-layer inputs and
MoE routing are outside this fixed-input bound.

No logits, NLL, PPL, or task accuracy was measured. In particular, neither
"1% operator error means 1% accuracy loss" nor "the small merge RMS proves
no quality loss" follows. A quality claim requires paired complete-model
evaluation with fixed weights/tokens and only the GDN implementation changed,
including A/A noise, teacher-forced delta NLL and KL, token-ranking changes,
long contexts, and task metrics.

## Portable reproduction on either PPU

The new entry point is
[`scripts/prefill/flashqla/repro_collinear_accuracy.py`](../scripts/prefill/flashqla/repro_collinear_accuracy.py).
It has a generation phase and a device-run phase. Its only Python runtime
dependency beyond the standard library is Torch. Building PLA also requires
the platform-compatible Torch/PPU SDK, a C++ build environment, setuptools,
ninja, and the pinned ACTLIZE headers.

From a checkout containing this PR, build only the relevant namespace in place:

```bash
git submodule update --init --recursive 3rdparty/actlize
python setup.py build_ext --inplace --namespace=pla.prefill.flashqla
```

The `run --repo .` identity check requires the Python source and loaded `_C`
extension to belong to that checkout. The in-place build supplies that
extension alongside the source; installing a wheel into site-packages alone
does not satisfy this checkout identity contract.

Set `PPU_SDK_DIR` or `PPU_ACTLIZE_DIR` when the corresponding dependencies
are outside their standard locations. The historical Torch 2.13 / SDK 2.2
image additionally needed the host-header shim described above; that is a
build compatibility detail, not an accuracy correction. A new build will
have its own binary hash and must not be labeled as the historical binary.

Run the CPU reference self-test, then generate fixtures once on CPU and copy
the same generated directory to both device environments:

```bash
python scripts/prefill/flashqla/repro_collinear_accuracy.py selftest

python scripts/prefill/flashqla/repro_collinear_accuracy.py generate \
  --output ./gdn-fixtures-quick --suite quick

python scripts/prefill/flashqla/repro_collinear_accuracy.py generate \
  --output ./gdn-fixtures-full --suite full
```

On an allocated 810E or 890P device exposed as logical device 0, the same run
command applies. Use a fresh output directory for each device/run:

```bash
python scripts/prefill/flashqla/repro_collinear_accuracy.py run \
  --fixtures ./gdn-fixtures-quick --output ./gdn-results-quick \
  --device cuda:0 --repo .

python scripts/prefill/flashqla/repro_collinear_accuracy.py run \
  --fixtures ./gdn-fixtures-full --output ./gdn-results-full \
  --device cuda:0 --repo .
```

For `run`, exit code 0 means all numerical and state-preservation checks
passed; 1 means at least one accuracy or state-preservation check failed;
2 means an execution, fixture, or identity error prevented a valid run.
Exit code 1 is expected when reproducing the documented accuracy failures;
inspect the recorded metrics and actual CP route rather than treating it
as an execution error.

The portable generator defaults to CPU RNG. The same seed does not make its
bytes equal to the historical CUDA-generated fixtures. Inspect and retain
the fixture hashes, reference method/dtype, loaded source and binary hashes,
actual CP selection, and empty-state result. The portable script tests the
same numerical contract and stress construction; it does not promise the
same numerical error or six identical threshold failures for different input
bytes. Byte-identical replay of the historical table requires the original
frozen fixtures and pinned binary, which are not shipped in this PR.

New-entry-point validation is recorded separately in
[portable_results_summary.json](gdn_collinear_accuracy/portable_results_summary.json).
It must not be used to relabel historical 810E evidence as a new portable
entry-point run. The historical same-binary 810E measurements remain the
810E evidence described above. Large fixtures, model weights, complete text
corpora, and output tensor dumps are intentionally outside the PR; the
checked-in JSON contains compact, sanitized evidence.

## New entry-point validation on 2026-11-12

The final portable CLI was run on ZW-M890P using the same pinned source archive
and dual-architecture binary recorded above. Each row below covers eight
fixtures, both public auto_cp settings, and 16 calls. Both long default calls
selected CP; the remaining calls selected non-CP. No execution exceptions
occurred, and both runs returned exit code 1 for the reproduced accuracy failures.

| Fixture set | Distinct stress failures / controls passing | Stress O RMS range | Nonempty state RMS range |
|---|---|---:|---:|
| Original frozen fixtures | 6/6 fail, 2/2 pass | 1.04065-1.17004% | 1.35836-1.65595% |
| Newly generated CPU fixtures | 6/6 fail, 2/2 pass | 1.02083-1.08566% | 1.26524-1.53782% |

All 32 O/nonempty-state maxima from original-fixture replay agree with the
historical 890P results within 1e-7 absolute ratio; the maximum delta is
1.59048e-8. Fixture hashes and actual CP selections match. The new comparator
reduces in FP64, whereas the historical comparator reduced in FP32.
The new CPU full suite also uses FP64 references for the 8192-token cases.

All four ragged calls in each new run failed empty-state preservation. This
is recorded separately and is consistent with an unwritten output buffer;
it does not change the nonempty stress findings or establish deterministic
empty-buffer contents.

The CPU selftest and full fixture generation passed. A separate read-only
review checked the state-layout adapter, FP64 formulas, CP observation,
nonfinite/empty-state handling, and source/binary identity guards. New 810E
execution was not attempted because the available host was occupied by an
inference workload with insufficient memory headroom. The historical 810E
same-input measurements remain valid historical evidence, not validation of
the new entry point on that device. No current-master kernel binary was
rebuilt for this documentation-and-diagnostics PR.

## Public model/source pins

- [Qwen3.8-Flash-Next config, revision 21a7c3ca68484a9987d62fd51b78130d9b9a11f8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/21a7c3ca68484a9987d62fd51b78130d9b9a11f8/config.json).
- [Qwen3.8-Max open checkpoint config, revision 912e40d903bfc891be8fe3c51faa0a09d182ce95](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/912e40d903bfc891be8fe3c51faa0a09d182ce95/config.json).
- [Max model and output tail, Transformers e4e4e98845fcf92e2b75f709f356fbc838b55d11](https://github.com/huggingface/transformers/blob/e4e4e98845fcf92e2b75f709f356fbc838b55d11/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
- [Next model, HC, and output tail, same Transformers revision](https://github.com/huggingface/transformers/blob/e4e4e98845fcf92e2b75f709f356fbc838b55d11/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
- [Gated DeltaNet paper](https://arxiv.org/html/2412.06464v3).
