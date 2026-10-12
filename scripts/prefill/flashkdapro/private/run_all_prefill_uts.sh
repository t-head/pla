#!/bin/bash
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

# Run the FlashKDA prefill building-block UTs against the installed
# pla.prefill.flashkdapro extension.
#
# These are the finer-grained checks kept under private/. The supported entry
# points live one level up:
#   scripts/prefill/flashkdapro/e2e_acc_full.py   end-to-end accuracy (all paths)
#   scripts/prefill/flashkdapro/bench_vs_fla.py   end-to-end performance vs FLA
#
# Usage (inside PPU1.5 container, from anywhere):
#   CUDA_VISIBLE_DEVICES=4 bash scripts/prefill/flashkdapro/private/run_all_prefill_uts.sh
#
# CANONICAL is self-contained: every check is against the fp32 torch reference in
# test_fwd_e2e.py, so it needs nothing beyond torch and the built extension.
#
# FLA_BASELINE additionally compares against the FLA Triton reference under
# scripts/prefill/private/ref. It remains opt-in via KDA_RUN_FLA=1 because it is
# a baseline comparison, not a correctness gate for this repo.
#
# NOTE: the scripts are run as plain scripts from this directory on purpose.
# Python then puts this directory on sys.path (so the shared torch reference
# resolves) while `import pla` still finds the *installed* package — putting the
# repo root on PYTHONPATH would shadow it with the uncompiled source tree.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONUNBUFFERED=1

CANONICAL=(
  test_bt64_contract.py    # H8 P1/H4 default gating, state layout and fallback contract (no GPU needed)
  test_fwd_e2e.py          # T=64/H=2 end-to-end vs fp32 torch reference
  test_fwd_cases.py        # 5 shapes incl. partial tail + full-DV and dvsplit paths
  test_kernel1.py          # Kernel 1 workspace (6 tensors) vs torch reference
)

TC01_CANONICAL=(
  test_cp_tc01_auto.py              # legacy tc01 AUTO CP + FMA-front boundaries and E2E state
  test_tc01_fla_monolithic_cp.py    # monolithic FLA CP splits, state pool and padding
)

FLA_BASELINE=(
  test_dispatch_paths.py   # all 5 dispatch paths vs FLA-Triton
)

run_one() {
  echo "======================================================================"
  echo ">>> RUN $1  (card=$CUDA_VISIBLE_DEVICES)"
  echo "======================================================================"
  python "$1"
}

CPASS=(); CFAIL=()
echo "############### CANONICAL (must pass) ###############"
for t in "${CANONICAL[@]}"; do
  if run_one "$t"; then echo "<<< $t EXIT 0"; CPASS+=("$t"); else echo "<<< $t EXIT $?"; CFAIL+=("$t"); fi
done

TCPASS=(); TCFAIL=(); TCSKIP=()
if python - <<'PY'
import torch
p = torch.cuda.get_device_properties(torch.cuda.current_device())
raise SystemExit(0 if (int(p.major), int(p.minor)) == (8, 0) else 1)
PY
then
  echo "############### TC01 CANONICAL (must pass on tc01) ###############"
  for t in "${TC01_CANONICAL[@]}"; do
    if run_one "$t"; then echo "<<< $t EXIT 0"; TCPASS+=("$t"); else echo "<<< $t EXIT $?"; TCFAIL+=("$t"); fi
  done
else
  TCSKIP=("${TC01_CANONICAL[@]}")
  echo "############### TC01 CANONICAL skipped (selected device is not tc01) ###############"
fi

FPASS=(); FFAIL=()
if [ "${KDA_RUN_FLA:-0}" = "1" ]; then
  echo "############### FLA BASELINE (opt-in) ###############"
  for t in "${FLA_BASELINE[@]}"; do
    if run_one "$t"; then echo "<<< $t EXIT 0"; FPASS+=("$t"); else echo "<<< $t EXIT $?"; FFAIL+=("$t"); fi
  done
else
  echo "############### FLA BASELINE skipped (set KDA_RUN_FLA=1) ###############"
fi

echo "######################################################################"
echo "CANONICAL PASS (${#CPASS[@]}/${#CANONICAL[@]}): ${CPASS[*]}"
echo "CANONICAL FAIL (${#CFAIL[@]}): ${CFAIL[*]}"
echo "TC01 CANONICAL PASS (${#TCPASS[@]}/${#TC01_CANONICAL[@]}): ${TCPASS[*]}"
echo "TC01 CANONICAL FAIL (${#TCFAIL[@]}): ${TCFAIL[*]}"
echo "TC01 CANONICAL SKIP (${#TCSKIP[@]}): ${TCSKIP[*]}"
if [ "${KDA_RUN_FLA:-0}" = "1" ]; then
  echo "FLA BASELINE PASS (${#FPASS[@]}/${#FLA_BASELINE[@]}): ${FPASS[*]}"
  echo "FLA BASELINE FAIL (${#FFAIL[@]}): ${FFAIL[*]}"
fi
echo "######################################################################"
[ ${#CFAIL[@]} -eq 0 ] && [ ${#TCFAIL[@]} -eq 0 ]
