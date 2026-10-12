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

# Wait for all 8 GPUs to be idle for 5 continuous minutes, then launch perf test.
# Usage: sudo docker exec <container> bash <repo>/scripts/decode/wait_and_run.sh

# Repo root, derived from this script's location (scripts/decode/ -> root).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
IDLE_THRESHOLD_UTIL=30    # GPU util % threshold (below = idle)
IDLE_THRESHOLD_MEM_PCT=30 # Memory % threshold (below = idle)
REQUIRED_IDLE_SECS=300    # 5 minutes
POLL_INTERVAL=30          # poll every 30 seconds

cd "$WORK_DIR" || exit 1

echo "=== GPU Idle Monitor ==="
echo "Waiting for all 8 GPUs to be idle (util<${IDLE_THRESHOLD_UTIL}%, mem<${IDLE_THRESHOLD_MEM_PCT}%) for ${REQUIRED_IDLE_SECS}s"
echo "Polling every ${POLL_INTERVAL}s..."
echo ""

first_idle_time=0
idle_count=0

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    # Query GPU status
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    all_idle=true
    busy_gpus=""
    while IFS=',' read -r idx util mem_used mem_total; do
        idx=$(echo "$idx" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        mem_total=$(echo "$mem_total" | tr -d ' ')
        mem_pct=$((mem_used * 100 / mem_total))

        if [ "$util" -ge "$IDLE_THRESHOLD_UTIL" ] || [ "$mem_pct" -ge "$IDLE_THRESHOLD_MEM_PCT" ]; then
            all_idle=false
            busy_gpus="${busy_gpus} GPU${idx}(util=${util}%,mem=${mem_pct}%)"
        fi
    done <<< "$gpu_info"

    if [ "$all_idle" = true ]; then
        if [ "$first_idle_time" -eq 0 ]; then
            first_idle_time=$(date +%s)
            echo "[$timestamp] All GPUs idle - starting 5min countdown..."
        fi
        elapsed=$(($(date +%s) - first_idle_time))
        remaining=$((REQUIRED_IDLE_SECS - elapsed))
        if [ "$elapsed" -ge "$REQUIRED_IDLE_SECS" ]; then
            echo "[$timestamp] All GPUs idle for ${REQUIRED_IDLE_SECS}s. Starting performance test!"
            break
        else
            echo "[$timestamp] All idle, ${remaining}s remaining..."
        fi
    else
        if [ "$first_idle_time" -ne 0 ]; then
            echo "[$timestamp] GPUs became busy again, resetting countdown."
        fi
        first_idle_time=0
        echo "[$timestamp] Busy:${busy_gpus}"
    fi

    sleep "$POLL_INTERVAL"
done

echo ""
echo "=== Starting Performance Test ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run the parallel performance comparison
bash scripts/decode/perf_compare_parallel.sh

echo ""
echo "=== Performance Test Complete ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
