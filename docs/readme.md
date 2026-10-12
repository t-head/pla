# PLA API Reference

PLA provides PyTorch extensions for linear-attention(GDN and KDA) decode and
prefill on PPU. The current build targets `ppu001` and `ppu0015`. Nothing is
exported from the root `pla` package; import the required namespace explicitly.

## Public surface

| Namespace and API | Scope | Verified framework integrations | Compiled artifact |
|---|---|---|---|
| `pla.decode.fused_sigmoid_gating_delta_rule_forward_k_last` | Non-packed GDN/KDA decode; single-chain and EAGLE-tree MTP | SGLang `v0.5.16`; `v0.5.18` for GLM-5.3-Flash; vLLM `v0.27.0` | `pla/decode/_C*.so` |
| `pla.decode.fused_sigmoid_gating_delta_rule_forward_k_last_packed` | Packed GDN/KDA decode | SGLang `v0.5.16`; vLLM `v0.27.0` | same decode extension |
| `pla.decode.kda.fused_recurrent_kda` | vLLM-specific recurrent KDA decode | Version not tracked | same decode extension |
| `pla.decode.kda.fused_kda_decode_mega_forward` | Fused Kimi K3 decode | SGLang `v0.5.17`; vLLM `v0.27.0` | same decode extension |
| `pla.decode.gdn.v_last.fused_sigmoid_gating_delta_rule` | v-last GDN decode wrapper | SGLang `v0.5.9` | `pla/decode/gdn/v_last/_C*.so` |
| `pla.decode.kda_mtp_sglang.fused_kda_decode_mtp_dspark` | Kimi K3 ReplaySSM verify | SGLang `v0.5.17` | `pla/decode/kda_mtp_sglang/_C*.so` |
| `pla.decode.kda_mtp_sglang.commit_kda_replayssm_after_verify` | Accepted-prefix state commit | SGLang `v0.5.17` | same ReplaySSM extension |
| `pla.prefill.flashqla.chunk_gated_delta_rule_fwd` | GDN prefill | vLLM `v0.27.0` | `pla/prefill/flashqla/_C*.so` |
| `pla.prefill.flashqla.gdn_prefill_packed_qkv_fwd` | Packed-QKV GDN prefill without a separate QKV split | Version not tracked | same FlashQLA extension |
| `pla.prefill.flashkdapro.flashkda_fwd` | KDA prefill | SGLang `v0.5.17`; vLLM `v0.27.0` | `pla/prefill/flashkdapro/_C*.so`; tc01 also uses internal `_tc01_fla/_C*.so` |

Namespaces named `private` expose test, tuning, or pipeline building blocks and
carry no compatibility guarantee.

## Build and install

Requirements:

- a PPU SDK; set `PPU_SDK_DIR` to its root if it is not installed at
  `/usr/local/PPU_SDK`;
- PyTorch available to the build process;
- ACTLIZE for both prefill namespaces, either in `3rdparty/actlize` or at
  `PPU_ACTLIZE_DIR`.

```bash
# Build all selectable namespaces with the production template set.
python scripts/build.py --mode tuned

# Build one selectable namespace.
python scripts/build.py --namespace pla.decode
python scripts/build.py --namespace pla.decode.gdn.v_last
python scripts/build.py --namespace pla.decode.kda_mtp_sglang
python scripts/build.py --namespace pla.prefill.flashqla
python scripts/build.py --namespace pla.prefill.flashkdapro

# Direct installation after generating pla.decode instantiations.
pip install . --no-build-isolation
```

The selectable namespaces and their supported public APIs are:

| Build namespace | Public API support set |
|---|---|
| `pla.decode` | `fused_sigmoid_gating_delta_rule_forward_k_last`; `fused_sigmoid_gating_delta_rule_forward_k_last_packed`; `pla.decode.kda.fused_recurrent_kda`; `pla.decode.kda.fused_kda_decode_mega_forward` |
| `pla.decode.gdn.v_last` | `fused_sigmoid_gating_delta_rule` |
| `pla.decode.kda_mtp_sglang` | `fused_kda_decode_mtp_dspark`; `commit_kda_replayssm_after_verify` |
| `pla.prefill.flashqla` | `chunk_gated_delta_rule_fwd`; `gdn_prefill_packed_qkv_fwd` |
| `pla.prefill.flashkdapro` | `flashkda_fwd` |

`PLA_BUILD_NAMESPACE` is the environment-variable equivalent of `--namespace`
and accepts comma-separated values. An exact namespace selects only that
extension; a non-exact prefix such as `pla.prefill` selects all matching
namespaces.

The source tree contains Python facades but not an importable in-tree replacement
for installed extensions. Run validation scripts as files and import the
installed package; do not add the repository root to `PYTHONPATH`.

## Decode auto-tuning

Run auto-tuning when a new GDN/KDA model introduces decode head configurations or
batch ranges not represented in the shipped lookup tables:

1. Add the model's real decode cases to `caselist/decode/`. Each record must carry
   the actual batch size, Q/K and value-head counts, head dimensions, and dtype.
2. Run the end-to-end tuner on an idle target PPU:

   ```bash
   bash scripts/decode/run_auto_tune.sh
   ```

   Add `--no-acu` for a faster timing pass, or use `--max-batch <N>` to cap the
   sampled batch range.
3. The pipeline builds all 102 decode template combinations, benchmarks
   `k_last`, `k_last_mtp`, and `k_last_packed`, writes a device-specific
   `lut/tuned_result_<device>.csv`, then rebuilds with the selected combinations.
4. Run the decode correctness tests and representative model workloads before
   publishing the updated lookup table. Subsequent production builds use
   `python scripts/build.py --mode tuned`.

For focused investigation, `auto_tune.py --case H,HV,K,V[,B] --kernel <name>`
tunes one shape and optionally one of `k_last`, `k_last_mtp`, or
`k_last_packed`. Use `--state-dtype bf16` when tuning the packed bf16-state path
and `--mtp-t <N>` to match the deployed MTP draft length. The automated sweep
currently measures the shared template geometry through the GDN path. For a KDA
model, validate candidate configurations explicitly before publishing them:

```bash
python scripts/decode/bench_single.py \
  --kernel k_last --case H,HV,128,128,B \
  --config BV,WN,SN,HV_PER_CTA --is-kda --check
```

## Common notation and state conventions

- `N`: sequence count; `T`: physical token count; `H`: Q/K heads; `HV`: value
  heads; `K`/`V`: QK/value head dimensions.
- Unless an API says otherwise, tensors must be on one PPU device.
- Decode kernels require `K == V == 128` and `HV` to be a positive multiple of
  `H`.
- K-last state uses `[slots, HV, V, K]`; v-last state uses
  `[slots, HV, K, V]`. State rows may have a larger slot pitch only where the
  API explicitly permits it.
- `cu_seqlens` describes packed sequences. Valid offsets start at zero, are
  monotonic, stay within physical token storage, and fit the int32 kernel ABI.
- SGLang uses negative slots as padding and permits slot `0`. vLLM treats slots
  `<= 0` as padding in the shared k-last kernels.
- Active requests that mutate state must not target the same state slot.

## Decode: k-last GDN and KDA

```python
from pla.decode import (
    fused_sigmoid_gating_delta_rule_forward_k_last,
    fused_sigmoid_gating_delta_rule_forward_k_last_packed,
)
```

### `fused_sigmoid_gating_delta_rule_forward_k_last`

UT-style usage from `scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py`:

```python
out = fused_sigmoid_gating_delta_rule_forward_k_last(
    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
    state, state_indices, 1.0, True, cu_seqlens, is_kda,
    False, None, None, None, None,
)
```

This entry supports non-packed GDN/KDA decode and single-chain or EAGLE-tree
MTP. It allocates an output with `v.shape` and updates the state pool unless the
selected mode suppresses the final commit.

Required contract:

- `q`, `k`: `[B,T,H,128]`; `v`: `[B,T,HV,128]`; dtype fp16 or bf16.
  Q/K/V head and feature axes are contiguous. A padded token stride is allowed.
- `a` and `b` have the same dtype and flattened token count as Q/K/V. `b` has
  width `HV`; `a` has width `HV` for GDN or `HV*128` for KDA. KDA also accepts
  the unflattened trailing shape `[HV,128]`.
- `A_log` is contiguous with `HV` fp32/fp16/bf16 elements.
- `dt_bias` is contiguous with `HV` elements for GDN or `HV*128` for KDA. Its
  dtype matches Q except on KDA MTP/verify, where it must be fp32.
- `softplus_beta` is positive; softplus parameters and `scale` are finite.
- `lower_bound`, when supplied, is finite, negative, and valid only for KDA.
- `initial_state_source` and `initial_state_indices` are required by the current
  launcher. The canonical state shape is `[slots,HV,128,128]`; fp32 and bf16 are
  supported. A flattened pool row is accepted when its innermost stride is one
  and `stride(0)` covers a complete state. Inner slot contents are contiguous.
- `initial_state_indices` is contiguous-column int32 `[N]`, or `[N,S]` for vLLM
  speculative decode. A 2-D table requires `is_sglang=False`.
- `cu_seqlens`, when supplied, is contiguous int32 `[N+1]`.

MTP mode is selected by intermediate-state metadata, an integer
`cache_steps_or_num_accept`, a parent table, or a 2-D slot table. For SGLang,
`cache_steps_or_num_accept` is a positive integer and the contiguous
`intermediate_states_buffer` shares the state dtype. For vLLM, it is contiguous
int32 `[N]` accepted-token counts used with a 2-D slot table. A parent table is
int32/int64 `[N, >= token_capacity]` and is valid only with the SGLang
intermediate cache.

`is_sglang=None` resolves to `True`; pass the framework convention explicitly in
integration code.

### `fused_sigmoid_gating_delta_rule_forward_k_last_packed`

UT-style usage from
`scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py`:

```python
out = fused_sigmoid_gating_delta_rule_forward_k_last_packed(
    mixed_qkv, a, b, A_log, dt_bias, 1.0, 20.0, K ** -0.5,
    state, state_indices, out, True, cu_seqlens, is_kda,
)
```

This is the packed, one-token-per-request path for both GDN and KDA. `mixed_qkv` is
`[tokens, 2*H*128 + HV*128]`, fp16 or bf16, with contiguous feature rows and a
possibly padded token stride. The state is `[slots,HV,128,128]` fp32/bf16 with
contiguous inner contents and a 16-byte-aligned slot pitch. `out` is contiguous
`[tokens,1,HV,128]` and shares the activation dtype. State indices are contiguous
int32 `[N]`; every `cu_seqlens` interval is zero or one. This entry point does not
accept `lower_bound`.

## Decode: KDA compatibility APIs

```python
from pla.decode.kda import fused_recurrent_kda, fused_kda_decode_mega_forward
```

### `fused_recurrent_kda`

This vLLM-specific recurrent KDA wrapper consumes raw gate and beta inputs for
speculative decode and returns `(out, initial_state)`, where `initial_state` is
updated in place.

- `q`, `k`, `v`, `raw_g`: `[1,T,H,128]`; Q/K/V are matching fp16 or bf16.
- `raw_beta`: `[1,T,H]`; `A_log`: `H` fp32/fp16/bf16 values.
- `dt_bias` is required and must contain `H*128` fp32 values.
- `initial_state`: `[slots,H,128,128]` fp32.
- `cu_seqlens`, `ssm_state_indices`, and optional `num_accepted_tokens` accept
  int32/int64 metadata whose values must fit int32. A 2-D slot table is the vLLM
  layout and requires `is_sglang=False`.
- `out`, when supplied, matches `v` in shape and dtype.
- `fuse_gate` must be `True` or `None`; `lower_bound`, when supplied, is finite
  and negative. `is_sglang` defaults to `False`.

UT-style usage from `scripts/decode/test_kda_decode.py`:

```python
out, final_state = fused_recurrent_kda(
    q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
    A_log=A_log, dt_bias=dt_bias, lower_bound=None,
    initial_state=state, cu_seqlens=cu_seqlens,
    ssm_state_indices=state_indices,
    num_accepted_tokens=num_accepted_tokens, out=out_buffer,
    is_sglang=is_sglang,
)
```

### `fused_kda_decode_mega_forward`

This Kimi K3 fused decode API is integrated by SGLang and vLLM. It
performs causal convolution, gating, recurrence, and optional output RMSNorm,
writes `conv_state`, `state`, and `out` in place, and returns `None`.

- `x`: `[B,3*H*128]` fp16/bf16, channel-contiguous;
  `H in {6,12,24,48,96}`.
- `weight`: contiguous fp32 `[3,4,H*128]`; optional `bias`: contiguous fp32
  `[3*H*128]`.
- `conv_state`: activation dtype `[slots,3*H*128,3]` in the SD cache layout.
- `raw_g`: contiguous `[1,B,H,128]`; `raw_beta`: `[1,B,H]` with contiguous heads;
  both share `x.dtype`.
- `a_log`: contiguous fp32 `[H]`; `dt_bias`: contiguous fp32 `[H,128]` or an
  equivalent flat buffer.
- `state_indices`: contiguous int32 `[B]`; `state`: fp32
  `[slots,H,128,128]` with contiguous inner slot contents; `out`: contiguous
  activation dtype `[1,B,H,128]`.
- `output_gate` and `norm_weight` are supplied together. The gate is activation
  dtype `[B,H,128]` or `[1,B,H,128]`; the weight is contiguous fp32 `[128]`;
  `norm_eps` is finite and non-negative.

UT-style usage from `scripts/decode/kda_fusion/test_pad_slot.py`:

```python
fused_kda_decode_mega_forward(
    x=packed_x, weight=fused_weight, bias=None,
    conv_state=conv_state, raw_g=raw_g, raw_beta=raw_beta,
    a_log=A_log, dt_bias=dt_bias, state_indices=state_indices,
    state=state, out=out, lower_bound=lower_bound,
    output_gate=output_gate, norm_weight=norm_weight, norm_eps=1e-5,
)
```

`KDA_DECODE_IMPL=ref` selects the bf16-only reference implementation for A/B
validation.

## Decode: v-last GDN

```python
from pla.decode.gdn.v_last import (
    fused_sigmoid_gating_delta_rule,
    fused_sigmoid_gating_delta_rule_forward,
)
```

### `fused_sigmoid_gating_delta_rule`

The wrapper is forward-only and writes the caller-provided `o` and state pool.
Callers pass `T, B, H, HV, K, V` explicitly; these values must match the tensor
shapes.

- `K == V == 128`, `HV % H == 0`, and `H in {1,2,4,8,16,32}`.
- Q/K are `[batch,tokens,H,128]`; V is `[batch,tokens,HV,128]`.
- Q/K/V, `a`, `b`, `dt_bias`, and `o` are contiguous and share fp16 or bf16.
  `A_log` is contiguous fp32/fp16/bf16.
- `h0_source` is contiguous fp32 `[slots,HV,128,128]` in v-last order
  `[slots,HV,K,V]`; `h0_indices` is int32 and contains at least `B` entries.
- `use_initial_state=True` is required. Fixed-length decode requires `T == 1`.
  Varlen mode requires contiguous int32 `cu_seqlens [B+1]`, with each sequence
  containing zero or one token.

UT-style wrapper usage from
`scripts/decode/test_fused_sigmoid_gating_recurrent.py`:

```python
out = fused_sigmoid_gating_delta_rule(
    A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
    o=out_buffer, h0_source=state, h0_indices=state_indices,
    cu_seqlens=cu_seqlens, T=T, B=B, H=H, HV=HV, K=K, V=V,
    use_initial_state=True, use_qk_l2norm_in_kernel=True,
    is_varlen=cu_seqlens is not None,
)
```

### `fused_sigmoid_gating_delta_rule_forward`

This positional native entry underlies the wrapper. The same UT exercises it
through `fused_sigmoid_gating_delta_rule`; direct integration follows the wrapper
argument order:

```python
fused_sigmoid_gating_delta_rule_forward(
    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
    state, out_buffer, state_indices, cu_seqlens,
    scale, T, B, H, HV, K, V, True, True, cu_seqlens is not None,
)
```

Dispatch between the small- and big-batch paths is automatic.

## Decode: SGLang KDA MTP

```python
from pla.decode.kda_mtp_sglang import (
    fused_kda_decode_mtp_dspark,
    commit_kda_replayssm_after_verify,
)
```

Both functions are keyword-only and match the SGLang integration signatures;
they are integrated by SGLang.

### `fused_kda_decode_mtp_dspark`

This API implements CACHE_RING verify only. All four
`replayssm_*` tensors are required; `intermediate_ssm` is not consumed, and
`onorm_*` is unsupported. Its contract is:

- `x_q/x_k/x_v/g`: bf16 `[1,T,H,128]`; `beta`: bf16 `[1,T,H]`. Head/channel
  dimensions are contiguous; token strides may be padded.
- `w_q/w_k/w_v`: fp32 `[H*128,4]`; `A_log`: fp32 `[H]`; `dt_bias`: fp32
  `[H,128]`. The wrapper materializes contiguous views before native dispatch.
- recurrent state: contiguous fp32 `[slots,H,128,128]`; conv states:
  bf16 `[slots,H*128,3]`; conv windows: bf16
  `[scratch,steps,H*128,3]`. Conv tensors may use positive-stride SGLang views.
- raw-V/raw-K rings are contiguous bf16 `[slots,H,L,128]`; gate ring is
  contiguous fp32 with the same shape; beta ring is contiguous fp32
  `[slots,H,L]`; `L >= T/N`.
- output is contiguous bf16 `[1,T,H,128]`; index vectors and `cu_seqlens` are
  int32/int64. Every request uses the same dense verify window.
- `scale` is finite; `lower_bound` is finite and negative.

Verify leaves persistent recurrent and conv states unchanged and writes output,
ring caches, and conv-window snapshots.

UT-style usage from `scripts/decode/kda_mtp_sglang/test_correctness.py`:

```python
out = fused_kda_decode_mtp_dspark(
    x_q=x_q, x_k=x_k, x_v=x_v, w_q=w_q, w_k=w_k, w_v=w_v,
    cs_q=cs_q, cs_k=cs_k, cs_v=cs_v, g=g, beta=beta,
    A_log=A_log, dt_bias=dt_bias, recurrent_state=recurrent_state,
    intermediate_state_indices=intermediate_state_indices,
    intermediate_conv_q=intermediate_conv_q,
    intermediate_conv_k=intermediate_conv_k,
    intermediate_conv_v=intermediate_conv_v,
    ssm_state_indices=ssm_state_indices, cu_seqlens=cu_seqlens,
    lower_bound=lower_bound, scale=scale,
    replayssm_rawv=rawv_ring, replayssm_rawk=rawk_ring,
    replayssm_g=gate_ring, replayssm_beta=beta_ring,
)
```

### `commit_kda_replayssm_after_verify`

This API folds accepted ring prefixes into the fp32 recurrent state and restores
conv state from the selected window. Ring length is `1..32`; state/rings are
contiguous; accepted lengths are in range; metadata is int32/int64 `[B]`.
`mamba_track_indices` and `mamba_steps_to_track` must be supplied together.
`conv_state_layout` is `"dim_width"` or `"width_dim"`.

UT-style usage from `scripts/decode/kda_mtp_sglang/test_correctness.py`:

```python
commit_kda_replayssm_after_verify(
    spec_state=spec_state,
    state_batch_indices=state_batch_indices,
    accept_lens=accept_lens,
    last_correct_step_indices=last_correct_step_indices,
    mamba_track_indices=mamba_track_indices,
    mamba_steps_to_track=mamba_steps_to_track,
    null_block_id=-1,
)
```

## Prefill: FlashQLA

This implementation draws on the algorithm in the official
[FlashQLA project](https://github.com/QwenLM/FlashQLA). We thank the project
contributors for making the algorithm and reference implementation available.

```python
from pla.prefill.flashqla import (
    chunk_gated_delta_rule_fwd,
    gdn_prefill_packed_qkv_fwd,
)
```

### `chunk_gated_delta_rule_fwd`

This GDN prefill API is integrated by vLLM. It returns
`(g_cum, A, output, None, final_state)`. The indexed state pool is updated in
place even when `output_final_state=False`.

UT-style usage from `scripts/prefill/flashqla/e2e_acc_full.py`:

```python
g_cum, A, output, _, final_state = chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale=scale, initial_state=initial_state,
    cu_seqlens=cu_seqlens, auto_cp=True,
)
```

- Q/K: `[1,T,Hg,128]`; V: `[1,T,H,128]`; shared fp16 or bf16 dtype.
- Supported `(H,Hg)` pairs are `(128,16)`, `(64,16)`, `(64,8)`, `(48,16)`,
  `(32,16)`, `(32,8)`, `(32,4)`, `(24,8)`, `(16,8)`, `(16,4)`, `(16,2)`,
  `(12,4)`, `(8,4)`, `(8,2)`, `(6,2)`, and `(4,2)`.
- Activated mode uses contiguous fp32 `g` and `beta` shaped `[1,T,H]`.
- Raw GDN mode supplies `raw_gate_a`, `raw_beta_b`, `A_log`, and `dt_bias`
  together and requires `normalize_qk=True`. Raw gate/beta are activation dtype
  `[T,H]` with contiguous heads; parameters are fp32 `[H]`.
- Activated-mode features are contiguous. Raw GDN mode permits padded token
  strides while keeping head/features contiguous.
- `cu_seqlens` is converted to contiguous int32 `[N+1]`. Its final offset may be
  below physical `T`; the output suffix is zero. `raw_cu_seqlens` is the matching
  trusted host tuple used by planning.
- Legacy state is contiguous `[num_states>=N,H,128,128]` in `[H,DK,DV]` order.
  Indexed state is contiguous `[slots,H,128,128]` in `[H,DV,DK]` order and uses
  contiguous int32 `initial_state_indices [N]`. State dtype is fp32 or bf16.
  `final_state_dtype` must match an incoming state and otherwise defaults to
  fp32.

### `gdn_prefill_packed_qkv_fwd`

This API accepts activation-dtype `packed_qkv [T,2*Hg*128 + H*128]`. Features are
contiguous; token rows may have trailing padding. It uses raw GDN mode and
otherwise follows the same state, offset, return, and CP contracts. Integrations
can use it to avoid a separate QKV-split operation.

UT-style usage from `scripts/prefill/flashqla/private/test_raw_gating.py`:

```python
g_cum, A, output, _, final_state = gdn_prefill_packed_qkv_fwd(
    packed_qkv, raw_gate_a, raw_beta_b, A_log, dt_bias, cu_seqlens,
    num_qk_heads=Hg, num_v_heads=H, scale=128 ** -0.5,
    initial_state=state, initial_state_indices=state_indices,
    raw_cu_seqlens=raw_offsets, auto_cp=True,
)
```

FlashQLA uses chunk size 32.

## Prefill: FlashKDAPro

FlashKDAPro draws on the algorithms in the official
[FlashKDA project](https://github.com/MoonshotAI/FlashKDA) and
[cuLA project](https://github.com/inclusionAI/cuLA), including cuLA's intra-card
CP design. We thank their contributors for the open algorithms and reference
implementations.

```python
from pla.prefill.flashkdapro import flashkda_fwd
```

### `flashkda_fwd`

This KDA prefill API is integrated by SGLang and vLLM. It writes
caller-owned `out` and optional state buffers, and returns `None`.

UT-style usage from `scripts/prefill/flashkdapro/e2e_kimi_k3.py`:

```python
flashkda_fwd(
    q, k, v, g, beta, A_log, dt_bias, out,
    scale, lower_bound, initial_state, final_state, cu_seqlens,
)
```

- `q/k/v/g`: contiguous bf16 `[T,H,128]`; `g` is the raw gate.
- `beta`: bf16 `[T,H]` with contiguous heads and an int32-addressable token
  stride; it contains raw pre-sigmoid values.
- `A_log`: contiguous fp32 `[H]`; `dt_bias`: contiguous fp32 `[H,128]`.
- `out`: contiguous bf16 `[T,H,128]`, with no overlap with V or beta.
- `scale` is finite; `lower_bound` is in `[-5,0]`.
- `cu_seqlens` is optional contiguous `[N+1]`. int32 is native; int64 is accepted
  after range checking and conversion. `raw_cu_seqlens` is an optional matching
  host tuple. A padded physical suffix is zeroed.
- `initial_state` and `final_state` are contiguous bf16/fp32
  `[N,H,128,128]` and must share a dtype when both are present. They may alias
  only as the exact same view.
- `state_pool [slots,H,128,128]` plus contiguous int32 `state_indices [N]`
  provides indexed in-place state. It is mutually exclusive with
  `initial_state`/`final_state`; active writable indices must be in range and
  unique.

FlashKDAPro uses chunk size 16. Runtime capability `8.0` selects the tc01 path and
`8.9` selects the tc02 path; other capabilities are rejected. On tc01,
`PLA_FLASHKDA_TC01_IMPL=fla|legacy` overrides automatic backend selection. The
tc01 monolithic FLA extension and any selected trusted-offset CP plan must be
loaded and warmed on the target device and stream before Graph capture.

## Validation commands

Run tests against the installed package on an idle PPU:

```bash
PLA_SANITY_CHECK=1 python -m pytest -q tests/test_sanity_check.py

python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_mtp.py
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py
python scripts/decode/test_gdn_noncontiguous_stride.py
python scripts/decode/test_kda_decode.py
python tests/test_kda_mtp_sglang.py

python scripts/prefill/flashqla/e2e_acc_full.py
python scripts/prefill/flashkdapro/e2e_acc_full.py
```

Caselist validation is available through `tests/test_op_forge.py`.

## Debug

### Environment variables

| Variable | Purpose |
|---|---|
| `PLA_SANITY_CHECK=1` | Enable additional input-contract diagnostics; value checks are skipped during Graph capture. |
| `PLA_BYPASS_EXEC=1` | Skip native execution for tracing; no numerical guarantee. Read at extension initialization. |
| `PLA_BUILD_NAMESPACE` | Select build namespaces. |
| `PPU_SDK_DIR`, `PPU_ACTLIZE_DIR` | Select the PPU SDK and ACTLIZE checkout. |
| `MAX_JOBS` | Control build parallelism. |
| `KDA_DECODE_IMPL=ref` | Select the KDA mega-decode reference implementation. |
| `PLA_FLASHKDA_TC01_IMPL` | Select the tc01 FlashKDAPro backend: `fla` or `legacy`. |
| `PLA_FLASHKDA_TC01_FLA_CP` | Select tc01 FLA CP: `auto`, `off`/`0`, or `1/2/3/4/5/8`. |
| `PPU_LIB_PERF_INSTRUMENT` | Emit instrumented ranges tagged with case metadata. |
| `PPU_LIB_SHOW_PARAMS` | Print replay metadata; capture-time records omit device offsets that cannot be read safely. |
| `FF_GEMM_REORDER` | Select the FlashQLA GEMM schedule at build time; production default is `1`. |

### Build modes

`--mode` affects only `pla.decode` template instantiation. Production builds use
`tuned`; the other modes support development and performance investigation:

| Mode | Meaning | Use case |
|---|---|---|
| `default` | Build only the default template combination. | Fast functional debugging when tuned coverage is unnecessary. |
| `full` | Build all 102 valid template combinations. | Auto-tuning, template comparison, or diagnosing a missing/incorrect tuned entry. |
| `tuned` | Build only combinations selected by `lut/tuned_result_*.csv`. | Production deployment and performance validation. |

```bash
python scripts/build.py --mode default
python scripts/build.py --mode full
```

### Input contract and diagnostics

The documented shapes, dtypes, layouts, devices, index ranges, and aliasing rules
are part of the API contract whether diagnostics are enabled or not. Performance
paths intentionally avoid device-to-host checks, so invalid metadata may
otherwise fail asynchronously or produce undefined results.

Set `PLA_SANITY_CHECK=1` for boundary diagnostics. It is read at every call and
adds checks for:

- strided tensors on the expected PPU device, with non-negative strides;
- complete shape, dtype, layout, alignment, and int32-addressability contracts;
- monotonic offsets, valid and non-conflicting state slots, and valid MTP tables;
- forbidden partial aliases between mutable outputs, states, and inputs;
- finite values in consumed Q/K/V, weights, and selected state rows. Raw
  sigmoid/softplus gate inputs may contain saturating infinities.

In eager execution, value checks may copy metadata or selected values to the
host and synchronize. During Graph capture, native checks are metadata-only and
perform no diagnostic device-to-host transfer. Python paths that require host
offsets must receive `raw_cu_seqlens` or use a metadata plan warmed before capture.

`PLA_BYPASS_EXEC=1` is a tracing aid, not a correctness mode. It is read when an
extension is initialized, skips native preprocessing and kernels, leaves caller
buffers unchanged, and provides no numerical guarantee for newly allocated
outputs. Bypass takes precedence over sanity checks.
