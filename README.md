# PLA — PPU Linear Attention Kernels

Linear-attention kernels for PPU1.5, organised into API namespaces:

| Namespace | Contents | Artifact |
|---|---|---|
| `pla.decode` | `fused_sigmoid_gating_delta_rule_forward_k_last`, `..._k_last_packed` | `pla/decode/_C*.so` |
| `pla.decode.gdn.v_last` | `fused_sigmoid_gating_delta_rule`, `..._forward` (v-last layout) | `pla/decode/gdn/v_last/_C*.so` |
| `pla.decode.kda` | `fused_recurrent_kda`, `fused_kda_decode_mega_forward` | ″ |
| `pla.decode.private` | `..._k_last_tunable`, `..._k_last_packed_tunable` (auto-tune only) | ″ |
| `pla.prefill.flashqla` | `chunk_gated_delta_rule_fwd` | `pla/prefill/flashqla/_C*.so` |
| `pla.prefill.flashqla.private` | `fused_fwd`, `chunk_local_cumsum`, `get_warmup_chunks`, `kkt_solve`, `prepare_h`, `correct_h0`, `calc_cp_seqs` | ″ |
| `pla.prefill.flashkdapro` | `flashkda_fwd` | `pla/prefill/flashkdapro/_C*.so` |
| `pla.prefill.flashkdapro.private` | `flashkda_prepare` | ″ |

Full per-argument reference: [docs/readme.md](docs/readme.md).

`PLA_SANITY_CHECK=1` enables input diagnostics and can be toggled at runtime; in eager mode value checks may synchronize,
while under CUDA Graph capture only metadata is checked, with no diagnostic D2H.

Set `PLA_BYPASS_EXEC=1` before starting the process to skip the preprocessing and kernels inside the native APIs;
only the exact value `1` enables it, each extension reads it once at initialization, and changes after loading have no effect.
Existing out/state/cache are left unchanged; required new outputs are only allocated as uninitialized Tensors, with no numerical guarantee.
Python preprocessing and diversion still run, so neither zero kernels for the whole Python call nor a runnable model is guaranteed.
Also setting `PPU_LIB_SHOW_PARAMS=1` collects cases; device offsets are not read during capture,
and the corresponding records are explicitly marked incomplete.

Source layout:

```
src/common.cuh          host-only shared helpers (instrumentation / case tags)
src/pla_device_helpers.cuh  torch-free PPU device helpers
src/decode/k-last/      k-last decode kernels + generated/ template instantiations
src/decode/v-last/      v-last decode kernels (own extension, per-H instantiation TUs)
src/prefill/flashqla/    FlashQLA prefill kernels
src/prefill/flashkdapro/    FlashKDA (Kimi Delta Attention) prefill kernels
3rdparty/actlize/       ACTLIZE PPU CUTLASS adaptation (shared by both prefill exts)
pla/                    Python package (namespace facades over the .so files)
lut/                    tuned_result_<device>.csv auto-tune lookup tables
scripts/decode/         decode build/tune/bench/test scripts
scripts/prefill/flashqla/  FlashQLA prefill UTs
scripts/prefill/flashkdapro/  FlashKDA prefill E2E accuracy + performance entry points
tests/                  OpForge caselist-driven correctness + perf harness
```

## 1. Build and Install

The prefill namespaces need ACTLIZE v1.0.0, the PPU adaptation of CUTLASS 3.6.0
carried in `3rdparty/actlize` (see [docs/prefill_readme.md](docs/prefill_readme.md)):

```bash
git submodule update --init --recursive
# or point at an existing checkout:
export PPU_ACTLIZE_DIR=/path/to/actlize
```

### Build script (generate + install)
```bash
python scripts/build.py --mode default   # fastest: one default kernel config
python scripts/build.py --mode tuned     # uses lut/tuned_result_*.csv
python scripts/build.py --mode full      # all combos (required before auto-tune)
```

### Build one namespace only
```bash
python scripts/build.py --namespace pla.decode              # k-last only
python scripts/build.py --namespace pla.decode.gdn.v_last   # v-last only
python scripts/build.py --namespace pla.prefill

# equivalently, straight through setup.py
PLA_BUILD_NAMESPACE=pla.decode pip install .
python setup.py build_ext --inplace --namespace=pla.prefill
```

Default (no filter) builds all four namespaces into four separate `.so` files.
An exact namespace name selects just that one, so `pla.decode` does not drag in
the nested `pla.decode.gdn.v_last`; a token that is not itself a namespace
prefix-matches (`pla.prefill` selects both `pla.prefill.flashqla` and
`pla.prefill.flashkdapro`).

Restricting the namespace is the biggest build-time lever: a full k-last build
compiles 200+ generated translation units. The v-last kernels are instantiated
one TU per H value — as a single TU each took ~25 min and dominated the build.

### Manual (generate + pip)
```bash
python scripts/decode/generate_kernel_instantiations.py --default-only
pip install .

# tuned
python scripts/decode/generate_kernel_instantiations.py --tuned-csv 'lut/tuned_result_*.csv'
pip install .

# full
python scripts/decode/generate_kernel_instantiations.py
pip install .
```

Build-speed knobs: `MAX_JOBS` (ninja parallelism, defaults to core count),
`HGCC_THREADS` (per-file hgcc threads, default 16).

## 2. Testing

### Decode — k_last kernel
```bash
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py --batch 64 --tp 1 --qwen35 --dtype bf16
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py --benchmark
```

### Decode — k_last_mtp (multi-token prediction)
```bash
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_mtp.py
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_mtp.py --dtype bf16 --alogtype fp32
```

### Decode — k_last_packed (packed QKV)
```bash
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py --benchmark
```

### Decode — KDA
```bash
# pla.decode.kda.fused_recurrent_kda vs vendored PyTorch reference
SAIL_PLA_CUDA=1 python scripts/decode/test_kda_decode.py

# fused mega decode kernel vs vLLM Triton reference
python scripts/decode/kda_fusion/compare_kda_decode_fusion.py
python scripts/decode/kda_fusion/test_pad_slot.py
```

### Decode — non-contiguous stride (Qwen3.5-27B compatibility)
```bash
python scripts/decode/test_gdn_noncontiguous_stride.py
```

### Prefill
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/prefill/flashqla/e2e_acc_full.py
CUDA_VISIBLE_DEVICES=4 python scripts/prefill/flashqla/bench_vs_fla.py
CUDA_VISIBLE_DEVICES=4 python scripts/prefill/flashkdapro/e2e_acc_full.py
```

### OpForge caselist harness
```bash
export SAIL_PLA_CUDA=1

# single case (append tp:N when copying a line out of a caselist)
python tests/test_op_forge.py --kernel k_last \
  --format='[GDN]:[D][14]gdn_attn:seqlen_q:1,head_dim_qk:128,head_dim_v:128,num_heads_qk:2,num_heads_v:8,hidden_act:silu,conv_kdim:4,batch_size:14,dtype:bf16,tp:1'

# whole caselist (or a directory of them)
python tests/test_op_forge.py --kernel k_last \
  --caselist=caselist/decode/Qwen3.5-Max_decode_tp8_kvlimit_200GB_model_len_128K_gdn_op.list
```

### Benchmark a specific case
```bash
# k_last decode with an explicit template config
python scripts/decode/bench_single.py --kernel k_last --case 16,64,128,128,64 --config 64,8,4,1

# k_last_mtp with T=4
python scripts/decode/bench_single.py --kernel k_last_mtp --case 16,64,128,128,64 --config 64,8,4,1 --mtp-T 4

# k_last_packed
python scripts/decode/bench_single.py --kernel k_last_packed --case 8,16,128,128,32 --config 64,8,4,1

# use tuned dispatch (omit --config)
python scripts/decode/bench_single.py --kernel k_last --case 16,64,128,128,64

# with correctness check against the Triton reference
python scripts/decode/bench_single.py --kernel k_last --case 16,32,128,128,64 --config 64,8,4,1 --check

# KDA mode
python scripts/decode/bench_single.py --kernel k_last --case 4,8,128,128,16 --config 64,8,4,1 --is-kda
```

Case format: `H,HV,K,V,B`. Config format: `BV,WN,SN,HV_PER_CTA`.

## 3. Auto-tune

### One-command full pipeline
```bash
# full build → auto-tune → regenerate with tuned combos → rebuild
CUDA_VISIBLE_DEVICES=0,4 bash scripts/decode/run_auto_tune.sh

# CUDA events instead of acu (faster, less precise)
CUDA_VISIBLE_DEVICES=0,4 bash scripts/decode/run_auto_tune.sh --no-acu

# limit batch range
CUDA_VISIBLE_DEVICES=0,4 bash scripts/decode/run_auto_tune.sh --max-batch 256
```

All extra arguments are forwarded to `auto_tune.py`. Output:
`lut/tuned_result_{device_name}.csv` (one per device type), consumed by
`generate_kernel_instantiations.py --tuned-csv` to emit the runtime dispatch table.

### Step-by-step (manual)
```bash
# Step 1: full build
python scripts/decode/generate_kernel_instantiations.py
pip install .

# Step 2: auto-tune
CUDA_VISIBLE_DEVICES=0,4 python scripts/decode/auto_tune.py --batch-interval 16 --max-batch 512

# Step 3: rebuild with tuned results
python scripts/decode/generate_kernel_instantiations.py --tuned-csv 'lut/tuned_result_*.csv'
pip install .
```

### Auto-tune options
```
--caselist-dir DIR     Caselist directory (default: caselist/decode)
--batch-interval N     Batch size sampling interval (default: 16)
--max-batch N          Max batch size (default: 512)
--num-warmup N         Warmup iterations (default: 3)
--num-iters N          Measurement iterations (default: 10)
--kernel TYPE          Tune only: k_last, k_last_mtp, or k_last_packed
--no-acu               Use CUDA events instead of the acu profiler
--case H,HV,K,V[,B]    Tune a single case (shows per-combo detail log)
```

Use `CUDA_VISIBLE_DEVICES` to select GPUs; multiple GPUs run shapes in parallel.

```bash
# single case, all kernels, acu cycles
CUDA_VISIBLE_DEVICES=0 python scripts/decode/auto_tune.py --case 16,64,128,128,64

# single kernel type
CUDA_VISIBLE_DEVICES=0 python scripts/decode/auto_tune.py --case 16,64,128,128,64 --kernel k_last_mtp

# single shape across a batch range
CUDA_VISIBLE_DEVICES=0 python scripts/decode/auto_tune.py --case 16,64,128,128 --max-batch 256
```

## 4. Performance comparison vs Triton

```bash
# one caselist
python scripts/decode/perf_compare.py --caselist caselist/decode/<file>.list

# all caselists in parallel across idle GPUs, then build the HTML report
bash scripts/decode/perf_compare_parallel.sh

# wait for the machine to go idle first, then run the above
bash scripts/decode/wait_and_run.sh
```

Reports land in `reports/` (see [reports/README.md](reports/README.md)).

## 5. Environment variables

| Variable | Effect |
|---|---|
| `SAIL_PLA_CUDA` | v-last variant selection (0/1 auto, 2 smallbatch, 3 bigbatch) |
| `PPU_LIB_PERF_INSTRUMENT` | emit NVTX ranges tagged with the caselist string |
| `PPU_LIB_SHOW_PARAMS` | print a replayable OperatorsForge `--format` tag; packed GDN/KDA calls include `cu_seqlens_q` and prefill APIs emit `[P]` records |
| `PLA_BYPASS_EXEC=1` | skip PLA computation and GPU preprocessing; return zero placeholders using runtime memory operations; combine with `PPU_LIB_SHOW_PARAMS=1` to collect cases |
| `PLA_SANITY_CHECK=1` | assert input contracts before launch; content checks skip D2H during capture; bypass takes precedence |
| `KDA_DECODE_IMPL=ref` | route `fused_kda_decode_mega_forward` to the ported NVIDIA reference |
| `PLA_BUILD_NAMESPACE` | build-time namespace filter (see §1) |
| `MAX_JOBS`, `HGCC_THREADS` | build parallelism |
| `PPU_SDK_DIR`, `PPU_ACTLIZE_DIR` | PPU toolchain and ACTLIZE header locations |
| `FF_GEMM_REORDER` | prefill GEMM schedule A/B (default 1 = production) |

Both instrumentation variables are safe to leave on under CUDA graph capture: reading
`cu_seqlens` back to the host is illegal mid-capture, so records emitted while a graph
is being captured omit `cu_seqlens_q` and describe the call by its shapes only.
