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

# Run the prefill UT scripts against the installed pla.prefill.flashqla extension.
#
# Internal extended acceptance suite. The public release gates remain
# ../e2e_acc_full.py and ../bench_vs_fla.py.
#
# NOTE: this used to list a per-kernel UT suite (utils/test_chunk_cumsum.py,
# test_e2e.py, ...) that was never vendored into this repo — every entry pointed
# at a missing file, so the script could only ever fail. It now runs the UTs that
# actually exist here: the 16 (H,Hg) dispatch shapes plus focused CP, packed
# multi-sequence, raw-gating and indexed-state coverage.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONUNBUFFERED=1

CANONICAL=(
  e2e_acc_full.py          # all dispatch shapes + CP + h0, vs fp32 torch reference
  private/cp_acceptance.py         # CP fallback=0/1, fp16/bf16, h0, tail
  private/multi_sequence_acceptance.py  # packed varlen vs serial and fp32
  private/test_raw_gating.py       # raw gating + packed QKV + indexed state
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

echo "######################################################################"
echo "CANONICAL PASS (${#CPASS[@]}/${#CANONICAL[@]}): ${CPASS[*]}"
echo "CANONICAL FAIL (${#CFAIL[@]}): ${CFAIL[*]}"
echo "######################################################################"
[ ${#CFAIL[@]} -eq 0 ]
