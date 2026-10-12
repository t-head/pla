#!/usr/bin/env python3
# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Sync the OperatorsForge FlashLinearAttn caselists into pla/caselist.

In CI the checkout component first pulls the OperatorsForge repo, then calls this script to
copy the .list files of ``FlashLinearAttn/GeneralizedUTLists/inference/{decode,prefill}``
into ``pla/caselist/{decode,prefill}``, so the tested caselists match the single source of the operator
inventory without keeping a long-lived redundant copy in the pla repo.

Scope:
  - decode: full copy of the OpForge side
  - prefill: copy only the Qwen3.7 lists (prefix controlled by --prefill-include)

The sync is an "overwriting copy", not a "mirror": it only writes/overwrites same-name files and never deletes
other caselists already on the pla side. pla/caselist/decode has models OpForge currently lacks (e.g.
Kimi-Linear-48B-A3B, Qwen3.5-Max); a mirror sync would delete them by mistake.

Usage::

    python scripts/sync_opforge_caselist.py --opforge-root /path/to/OperatorsForge
    python scripts/sync_opforge_caselist.py --opforge-root ... --dry-run
"""
from pathlib import Path
import argparse
import filecmp
import shutil
import sys

# Relative path of the caselists inside the OpForge repo
OPFORGE_SUBPATH = Path("FlashLinearAttn/GeneralizedUTLists/inference")

# prefill only runs the Qwen3.7 caselists
DEFAULT_PREFILL_INCLUDE = ("Qwen3.7",)

# Operator families pla does not implement. The files are still copied (keeping the caselist source complete),
# but are listed separately in the sync summary as a reminder that they should not enter the CI run list.
UNSUPPORTED_OP_MARKERS = ("mamba2", "lightning")


def _resolve_list_dir(opforge_root):
    """Locate the OpForge inference caselist directory; also accepts passing the subdirectory directly."""
    root = Path(opforge_root)
    for candidate in (root / OPFORGE_SUBPATH, root):
        if (candidate / "decode").is_dir() or (candidate / "prefill").is_dir():
            return candidate
    return None


def _wanted(name, includes):
    """Empty includes means everything; otherwise the file name must contain one of the prefix keywords."""
    if not includes:
        return True
    return any(key in name for key in includes)


def sync_stage(src_dir, dst_dir, includes, dry_run):
    """Sync one stage directory and return a stats dict."""
    stats = {"copied": 0, "updated": 0, "same": 0, "filtered": 0,
             "unsupported": [], "names": []}
    if not src_dir.is_dir():
        print(f"WARNING: source directory does not exist, skipping: {src_dir}")
        return stats
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_dir.glob("*.list")):
        if not _wanted(src.name, includes):
            stats["filtered"] += 1
            continue
        dst = dst_dir / src.name
        if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
            stats["same"] += 1
        elif dst.is_file():
            stats["updated"] += 1
            if not dry_run:
                shutil.copy2(src, dst)
        else:
            stats["copied"] += 1
            if not dry_run:
                shutil.copy2(src, dst)
        stats["names"].append(src.name)
        if any(marker in src.name for marker in UNSUPPORTED_OP_MARKERS):
            stats["unsupported"].append(src.name)
    return stats


def _report(stage, stats):
    print(f"  [{stage}] copied={stats['copied']} updated={stats['updated']} "
          f"unchanged={stats['same']} filtered_out={stats['filtered']}")
    if stats["unsupported"]:
        print(f"      NOTE: the following {len(stats['unsupported'])} lists belong to operator families pla does not implement"
              f" ({'/'.join(UNSUPPORTED_OP_MARKERS)}); synced but should not enter the CI run list:")
        for name in stats["unsupported"]:
            print(f"        - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync OperatorsForge FlashLinearAttn caselists into pla/caselist")
    parser.add_argument("--opforge-root", required=True,
                        help="OperatorsForge repo root (or pass .../GeneralizedUTLists/inference directly)")
    parser.add_argument("--caselist-root", default=None,
                        help="pla caselist directory (default <repo>/caselist)")
    parser.add_argument("--prefill-include", default=",".join(DEFAULT_PREFILL_INCLUDE),
                        help="prefill only syncs lists whose file name contains one of these keywords, comma separated; "
                             f"an empty string means everything (default {','.join(DEFAULT_PREFILL_INCLUDE)})")
    parser.add_argument("--decode-include", default="",
                        help="same-name filter for decode, comma separated; empty by default, i.e. full sync")
    parser.add_argument("--dry-run", action="store_true", help="only print the changes that would be made, without writing")
    args = parser.parse_args()

    src_root = _resolve_list_dir(args.opforge_root)
    if src_root is None:
        print(f"ERROR: {OPFORGE_SUBPATH}/(decode|prefill) not found under {args.opforge_root}")
        return 2

    caselist_root = (Path(args.caselist_root) if args.caselist_root
                     else Path(__file__).resolve().parent.parent / "caselist")
    prefill_include = tuple(k.strip() for k in args.prefill_include.split(",") if k.strip())
    decode_include = tuple(k.strip() for k in args.decode_include.split(",") if k.strip())

    print(f"*--- sync OpForge caselist ---*")
    print(f"  source: {src_root}")
    print(f"  target: {caselist_root}")
    print(f"  decode filter : {decode_include or '(all)'}")
    print(f"  prefill filter: {prefill_include or '(all)'}")
    if args.dry_run:
        print("  DRY-RUN: no files will be written")

    decode_stats = sync_stage(src_root / "decode", caselist_root / "decode",
                              decode_include, args.dry_run)
    prefill_stats = sync_stage(src_root / "prefill", caselist_root / "prefill",
                               prefill_include, args.dry_run)
    _report("decode", decode_stats)
    _report("prefill", prefill_stats)

    total = sum(len(s["names"]) for s in (decode_stats, prefill_stats))
    if total == 0:
        print("ERROR: no caselist was synced; check --opforge-root and the filters")
        return 2
    print(f"  total synced: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
