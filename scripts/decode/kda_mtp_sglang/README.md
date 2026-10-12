``scripts/decode/kda_mtp_sglang/`` -- correctness + latency scripts for pla.decode.kda_mtp_sglang.

The two Kimi-K3 KDA MTP speculative-decoding entries (``fused_kda_decode_mtp_dspark`` /
``commit_kda_replayssm_after_verify``) are taken from the installed ``pla`` package; the
JIT ``cuda_ext`` build path of sglang_replayssm_ppu is no longer used.

## Files

Top-level scripts (run directly by users):

- ``test_correctness.py`` -- checks both the verify and fold paths against an fp64 oracle,
  plus three side-effect assertions: padding, read-only and conv rollback; ends with ``RESULT: all PASS``
  or ``FAILED: <names>`` and ``sys.exit(0/1)``.
- ``test_verify_layouts.py`` -- natively reads sglang split QKV, the conv pool and dense
  scratch views, compares outputs, rings and conv windows byte for byte against the contiguous layout, checks padding,
  unused scratch, read-only state and CUDA graph replay, covering both default VSPLIT paths.
- ``test_commit_layouts.py`` -- byte-for-byte comparison of the two conv layouts of the public commit,
  covering partial/full acceptance, padding, tracking and CUDA graph replay.
- ``bench_latency.py`` -- A/B curve bs=1..64; ``--tag/--reps/--inner/--kind``;
  persisted to ``results/<tag>.json`` (relative to CWD).
- ``bench_fold_vs_triton.py`` -- isolated ssm fold: pla CUDA vs upstream Triton,
  both checked against fp64, plus a 69-layer latency comparison.
- ``bench_fold_mtp_cases.py`` -- sweeps the OperatorsForge Kimi-K3 MTP case matrix
  (seqlen_q in {3,4,5,8} x heads in {12,24,48,96} x bs 1..~600): pla CUDA vs
  sglang Triton fold; ends with geomean speedup + win_rate. Large bs adaptively reduces the layer count.

Private helpers (``private/``) -- imported only by the top-level scripts, not run directly by users:

- ``private/_common.py`` —— ``Cfg`` / ``build_verify_inputs`` /
  ``build_commit_inputs`` / ``SpecState`` duck type / ``gpu_banner``
  (builds the tensors both entries need, with shapes matching the K3 dspark contract).
- ``private/_reference.py`` -- fp64 oracle: ``verify_reference`` replays
  dspark verbatim, ``fold_reference`` replays the exact fold verbatim; ``score`` / ``fmt`` report
  max_abs / max_rel / rel_l2.
- ``private/_triton_fold.py`` —— vendored sglang Triton fold
  (``kda_replayssm_exact_fold_kernel`` + ``triton_fold_all_layers``),
  ``HAS_TRACK=False``, used only for A/B.
- ``private/_harness.py`` -- ``Timer`` with 128MB L2 flush + median/P10/P90;
  ``save`` / ``load`` / ``table`` / ``clocks``.

The top-level scripts find the helpers with ``sys.path.insert(0, os.path.join(HERE, "private"))``;
`private/` has no ``__init__.py`` (consistent with ``scripts/decode/kda_fusion/``,
it is not treated as a Python package).

## Run

Inside the PPU server container (with ``pip install pla`` done):

```
python scripts/decode/kda_mtp_sglang/test_correctness.py
python scripts/decode/kda_mtp_sglang/test_verify_layouts.py
python scripts/decode/kda_mtp_sglang/test_commit_layouts.py
python scripts/decode/kda_mtp_sglang/bench_latency.py --tag baseline --kind both
python scripts/decode/kda_mtp_sglang/bench_fold_vs_triton.py
python scripts/decode/kda_mtp_sglang/bench_fold_mtp_cases.py --max-cases 40
```

``bench_latency.py`` writes ``results/<tag>.json`` under CWD (including the device banner +
an nvidia-smi clocks snapshot); ``bench_fold_mtp_cases.py`` accepts ``--json <path>`` to persist
per-case details and the summary.

SGLang's conv pool is ``[layers, slots, width, dim]``. When calling the public commit,
pass ``conv_state_layout="width_dim"`` to read/write the transposed views of the pool and scratch directly;
the default ``"dim_width"`` keeps the original PLA layout.
