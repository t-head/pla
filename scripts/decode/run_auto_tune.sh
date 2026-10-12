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

# Full auto-tune pipeline: full build → auto-tune → regenerate with tuned results → rebuild
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,4 bash scripts/decode/run_auto_tune.sh              # acu cycles (precise, slower)
#   CUDA_VISIBLE_DEVICES=0,4 bash scripts/decode/run_auto_tune.sh --no-acu     # CUDA events (fast)
#   CUDA_VISIBLE_DEVICES=0,4 bash scripts/decode/run_auto_tune.sh --max-batch 256  # limit batch range
#
# All arguments are forwarded to auto_tune.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# scripts/decode/ -> repo root
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

echo "============================================"
echo "Step 1/4: Generate all kernel instantiations"
echo "============================================"
python3 scripts/decode/generate_kernel_instantiations.py
echo ""

echo "============================================"
echo "Step 2/4: Build with all kernel combos"
echo "============================================"
pip install .
echo ""

echo "============================================"
echo "Step 3/4: Auto-tune"
echo "============================================"
python3 scripts/decode/auto_tune.py --batch-interval 16 --max-batch 512 "$@"
echo ""

echo "============================================"
echo "Step 4/4: Regenerate with tuned results and rebuild"
echo "============================================"
python3 scripts/decode/generate_kernel_instantiations.py --tuned-csv 'lut/tuned_result_*.csv'
pip install .
echo ""

echo "============================================"
echo "Done. Tuned CSV files:"
ls -la lut/tuned_result_*.csv
echo ""
echo "Generated kernel files:"
ls src/decode/k-last/generated/*.cu | wc -l
echo ""
echo "Run tests to verify:"
echo "  python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py"
echo "  python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py"
echo "  python scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_mtp.py"
echo "============================================"
