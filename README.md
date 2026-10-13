# PLA

## Overview

PLA provides high-performance PyTorch extensions for linear-attention decode and
prefill workloads on T-Head PPU. It includes optimized kernels for Gated Delta
Networks (GDN) and Kimi Delta Attention (KDA), targeting the `ppu001` and
`ppu0015` architectures.

## Features

- GDN and KDA decode kernels with k-last, v-last, packed, and MTP paths.
- FlashQLA GDN prefill and FlashKDAPro KDA prefill kernels.
- Dedicated PyTorch namespaces that can be built independently.
- CUDA Graph-compatible execution paths and optional runtime diagnostics.
- Device-specific decode auto-tuning and production lookup tables.
- Integrations validated with SGLang and vLLM workloads.

## Release Notes

### v1.0.0-Initial Public Release

The first public release of PLA includes the production PPU kernels, Python
interfaces, build tooling, validation scripts, and tuned decode configurations.
See [CHANGELOG.md](CHANGELOG.md) for details.

## Quick Start

### Requirements

- T-Head PPU with `ppu001` or `ppu0015` support
- PPU SDK
- Python and PyTorch
- [ACTLIZE](https://github.com/t-head/actlize) v1.0.0 for prefill kernels

Set `PPU_SDK_DIR` if the SDK is not installed at `/usr/local/PPU_SDK`. Set
`PPU_ACTLIZE_DIR` to use an ACTLIZE checkout outside `3rdparty/actlize`.

### Build and Install

```bash
git clone --recursive https://github.com/t-head/pla.git
cd pla

# Build all production namespaces.
python scripts/build.py --mode tuned

# Install the package.
pip install . --no-build-isolation
```

Build a single namespace when only one kernel family is required:

```bash
python scripts/build.py --namespace pla.decode
python scripts/build.py --namespace pla.decode.gdn.v_last
python scripts/build.py --namespace pla.decode.kda_mtp_sglang
python scripts/build.py --namespace pla.prefill.flashqla
python scripts/build.py --namespace pla.prefill.flashkdapro
```

## Interfaces

| Namespace | Capability |
|---|---|
| `pla.decode` | k-last GDN/KDA decode, packed decode, and KDA compatibility APIs |
| `pla.decode.gdn.v_last` | v-last GDN decode |
| `pla.decode.kda_mtp_sglang` | KDA ReplaySSM verification and state commit |
| `pla.prefill.flashqla` | FlashQLA GDN prefill and packed-QKV GDN prefill |
| `pla.prefill.flashkdapro` | FlashKDAPro KDA prefill |

For API signatures, tensor contracts, environment variables, and detailed usage,
see the [PLA API Reference](docs/readme.md). Prefill implementation details are
available in the [prefill guide](docs/prefill_readme.md).

## Validation

Run tests against the installed package on an idle PPU:

```bash
PLA_SANITY_CHECK=1 python -m pytest -q tests/test_sanity_check.py
python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py
python scripts/prefill/flashqla/e2e_acc_full.py
python scripts/prefill/flashkdapro/e2e_acc_full.py
```

## License

PLA is released under the [BSD 3-Clause License](LICENSE). Third-party software
and attribution information are listed in [NOTICE](NOTICE) and `LICENSES/`.
