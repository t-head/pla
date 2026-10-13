# Changelog

All notable changes to PLA are documented in this file.

## v1.0.0-Initial Public Release

- Added optimized GDN and KDA decode kernels for k-last, v-last, packed, and MTP workloads.
- Added FlashQLA GDN prefill and FlashKDAPro KDA prefill implementations.
- Added independent PyTorch extension namespaces for selective builds and deployment.
- Added production build tooling, device-specific decode tuning tables, and auto-tuning utilities.
- Added validation scripts for decode, prefill, input contracts, and representative framework integrations.
- Added support for the `ppu001` and `ppu0015` PPU architectures.
- Added API, build, debugging, and licensing documentation for public distribution.
