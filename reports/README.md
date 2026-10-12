# PLA CUDA vs Triton Performance Comparison Report

## Overview

This directory contains performance comparison data for the PLA (PPU Linear Attention) kernels between the CUDA and Triton implementations.
The tests cover every case in `caselist/decode/*.list`, spanning several models (Qwen3.5-Max, Qwen3.7-Max, Qwen3.5-397B-A17B, Qwen3.5-35B-A3B, Qwen3.5-27B, Qwen3-Next-80B-A3B, Kimi-Linear-48B-A3B, GLM-5.3-Flash, Nemotron-3-Super-120B-A12B) in decoding scenarios under different TP configurations.

### Test environment

- **Node A (primary)**: 30.70.218.2 (docker: pla_890p_perf), ZW-M890P × 8 (PPU1.5, compute capability 8.9)
- **Node B**: 11.165.159.10 (docker: hpwu-aiu-tc01), ZW810E × 16 (PPU1.0/1.1)
- **PyTorch**: 2.10.0, CUDA 13.0
- **Python**: 3.12.3
- **Test date**: 2026-09-14 (full refresh) + 2026-09-15 (GLM-5.3-Flash decode increment)
- **Harness methodology**: the state/out clone has been moved out of the timed region (consistent with the sglang production path), events mode; borderline losing cases were re-measured and confirmed with warmup=10
- **Known blocker**: Kimi-K3 decode (H=6/12/24/48/96, non-power-of-2 head counts) triggers out-of-bounds memory accesses in the pla-style kernels; none of the three kernels produces trustworthy data, so they are not included in the report. The mega kernel path works fine; to be re-measured after the kernel fix (see "Key conclusions" below)

### Metrics

Each case measures time along three dimensions:

| Metric | Description |
|------|------|
| `cuda_time_us` / `triton_time_us` | Device kernel time (CUDA events) |
| `cuda_host_us` / `triton_host_us` | Host time (time for the async call to return; pure host overhead) |
| `cuda_e2e_us` / `triton_e2e_us` | End-to-end time (host + device + sync) |
| `cycle_ratio` | triton_time / cuda_time (>1 means CUDA is faster) |

## Files

### Data files

| File | Rows | Description |
|------|------|------|
| `perf_k_last.csv` | 5678 | k_last kernel (standard decode, T=1), ZW-M890P |
| `perf_k_last_mtp.csv` | 5294 | k_last_mtp kernel (MTP verify, T=4), ZW-M890P |
| `perf_k_last_packed.csv` | 5678 | k_last_packed kernel (packed QKV), ZW-M890P |
| `perf_k_last_810e.csv` | 5673 | k_last, ZW810E |
| `perf_k_last_mtp_810e.csv` | 5289 | k_last_mtp, ZW810E |
| `perf_k_last_packed_810e.csv` | 5673 | k_last_packed, ZW810E |
| `triton_wins.list` / `triton_wins_810e.list` | 0 | List of cases where Triton beats CUDA (empty this round) |
| `perf_report_890p.html` / `perf_report_810e.html` | - | Per-device HTML reports |
| `perf_report_prefill.html` | - | GDN/KDA prefill acceptance and historical experiments (including the 2026-09-14 dual-device section) |

### Scripts

| File | Purpose |
|------|------|
| `scripts/decode/perf_compare_parallel.sh` | Main entry: auto-detects idle GPUs, dispatches caselists in parallel, merges results |
| `scripts/decode/perf_compare.py` | Single-GPU worker: iterates over the cases in a caselist and calls the runner for CUDA/Triton |
| `scripts/decode/_perf_compare_runner.py` | Single-case runner: builds tensors, runs the kernel, measures device/host/e2e time |
| `scripts/decode/wait_and_run.sh` | Waits until the GPUs have been idle for 5 minutes, then starts the performance test |
| `scripts/decode/gen_decode_report.py` | Generates a self-contained HTML report from the CSVs (filterable by model/TP/kernel; `--device` labels the device) |

## Usage

### Run the full performance comparison

```bash
# Auto-detect idle GPUs and run (util < 30% counts as idle by default)
bash scripts/decode/perf_compare_parallel.sh

# Custom memory limit and utilization threshold
bash scripts/decode/perf_compare_parallel.sh 8 50

# Specify GPUs manually
CUDA_VISIBLE_DEVICES=0,2,5 bash scripts/decode/perf_compare_parallel.sh
```

### Generate the HTML report separately

```bash
# 890P main report (the CSVs under reports/ are the 890P ones)
python3 scripts/decode/gen_decode_report.py --report-dir reports/ \
    --device "ZW-M890P (PPU1.5, 30.70.218.2)" --output reports/perf_report_890p.html

# 810E report (first link *_810e.csv under the standard file names into a separate directory)
python3 scripts/decode/gen_decode_report.py --report-dir <dir-with-std-names> \
    --device "ZW810E (PPU1.0/1.1, 11.165.159.10)" --output reports/perf_report_810e.html
```

### View the report

Open `perf_report_890p.html` or `perf_report_810e.html` directly in a browser (no web server needed):

```
file:///path/to/reports/perf_report_890p.html
```

## Key conclusions (2026-09-14 full sweep + 2026-09-15 GLM-5.3-Flash increment)

1. **CUDA beats Triton across the board**: full sweep of 5673+ cases per device (26 caselists), triton-win is 0 on both (on 890P the initial GLM-5.3-Flash sweep showed 5 borderline cases; re-measured with warmup=10 on a clean card they all flipped to CUDA wins, and the re-measured rows have been merged back into the main CSV)
2. **ZW-M890P (PPU1.5)**: k_last median 1.54x (min 1.07), k_last_mtp median 1.45x (min 1.29), k_last_packed median 1.32x (min 1.02)
3. **ZW810E (PPU1.0/1.1)**: k_last median 1.45x (min 1.01), k_last_mtp median 1.28x (min 1.01), k_last_packed median 1.37x (min 1.00)
4. **GLM-5.3-Flash (KDA)**: all 1165 cases per kernel for tp1/tp2/tp4/tp8/tp16 on both devices are recorded; CUDA wins every case (e.g. 890P k_last median about 1.6~2.3x)
5. **Kimi-K3 (KDA) not recorded**: non-power-of-2 head counts (H=6/12/24/48/96) trigger out-of-bounds memory accesses in the pla-style decode kernels -- k_last/k_last_mtp crash deterministically, k_last_packed crashes non-deterministically (the same case may pass or fail); on 890P the crash also puts the GPU into a driver-level unreliable state (requires `ppu-smi -r -i <id>` reset). The mega kernel (sglang interface) works fine on the same shapes; to be re-measured after the kernel fix
6. **Prefill**: full replay of the 26 caselists on 890P gives a median speedup of about 2.0~2.3x (both GDN and KDA within tolerance); on 810E GDN median is about 1.2x, KDA only about 0.05x (about 20x slower than FLA-Triton, suspected to be a build missing the bf16 hardware conversion instruction and falling back to scalar emulation; under investigation)

### Covered models

- Qwen3.5-Max (tp1/tp2/tp4/tp8)
- Qwen3.7-Max (tp1/tp2/tp4/tp8)
- Qwen3.5-397B-A17B (tp1/tp2/tp4/tp8)
- Qwen3.5-35B-A3B (tp1/tp2)
- Qwen3.5-27B (tp1/tp2)
- Qwen3-Next-80B-A3B (tp1/tp2/tp4)
- Kimi-Linear-48B-A3B (tp1/tp2)
- GLM-5.3-Flash (tp1/tp2/tp4/tp8/tp16)
- Nemotron-3-Super-120B-A12B (tp1/tp2/tp4/tp8)
- Kimi-K3 (tp1/tp2/tp4/tp8/tp16) -- caselist is checked in, no data yet due to the kernel OOB defect (see above)
