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

"""Regression test: the case dump must survive CUDA graph capture.

``PPU_LIB_PERF_INSTRUMENT=1`` / ``PPU_LIB_SHOW_PARAMS=1`` make every decode entry
point serialize an OperatorsForge case string before launching, and that
serializer reads ``cu_seqlens`` back to the host. Serving stacks capture the
decode step into a graph, and torch rejects unpinned D2H copies mid-capture:

    RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
    capture unless the CPU tensor is pinned.

so a model that ran fine with the instrumentation off died the moment it was
turned on (sglang ``gdn_backend`` target_verify -> ... ->
``fused_sigmoid_gating_delta_rule_forward_k_last``). Instrumentation has to stay
observational, hence this test.

The env vars are read once in C++ when the extension is imported
(``init_env_variables``), so a single process can only exercise one setting: the
driver re-execs itself once per configuration and reports each worker's status.

    python tests/test_instrument_graph_capture.py             # all configs
    python tests/test_instrument_graph_capture.py --worker perf
"""
import argparse
import os
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

# Run straight from a checkout: sys.path[0] is tests/, so the repo root (holding
# the in-place built pla/) has to be added explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def _load_device_dependencies():
    global torch, k_last
    import torch
    from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last as k_last

# Small on purpose: this is a host-side contract test, not a kernel benchmark.
B, H, HV, K, V = 2, 4, 8, 128, 128
DRAFT_TOKENS = 4

# One process per env setting; the worker name doubles as the --worker argument.
CONFIGS = (
    ("off", {"PLA_SANITY_CHECK": "0", "PPU_LIB_PERF_INSTRUMENT": "0", "PPU_LIB_SHOW_PARAMS": "0"},
     "control: capture works with the dump disabled"),
    ("perf", {"PLA_SANITY_CHECK": "0", "PPU_LIB_PERF_INSTRUMENT": "1", "PPU_LIB_SHOW_PARAMS": "0"},
     "the reported failure: NVTX tagging during capture"),
    ("show", {"PLA_SANITY_CHECK": "0", "PPU_LIB_PERF_INSTRUMENT": "0", "PPU_LIB_SHOW_PARAMS": "1"},
     "same D2H read reached through the params dump"),
    ("sanity", {"PLA_SANITY_CHECK": "1", "PPU_LIB_PERF_INSTRUMENT": "0",
                "PPU_LIB_SHOW_PARAMS": "0"},
     "input value checks are skipped during capture"),
    ("sanity_perf", {"PLA_SANITY_CHECK": "1", "PPU_LIB_PERF_INSTRUMENT": "1",
                     "PPU_LIB_SHOW_PARAMS": "0"},
     "sanity checks and performance instrumentation coexist"),
    ("sanity_show", {"PLA_SANITY_CHECK": "1", "PPU_LIB_PERF_INSTRUMENT": "0",
                     "PPU_LIB_SHOW_PARAMS": "1"},
     "sanity checks and parameter dumps coexist"),
)


def _make_inputs(tokens_per_request, empty_last=False):
    """Decode-shaped k-last inputs, packed as the frameworks pass them.

    tokens_per_request == 1 is a plain decode step; > 1 is an MTP verify step,
    which is what the reported traceback was in.
    """
    torch.manual_seed(2026)
    device = "cuda"
    active_requests = B - 1 if empty_last else B
    total = active_requests * tokens_per_request
    inputs = {
        "A_log": torch.randn(HV, dtype=torch.float32, device=device),
        "dt_bias": torch.randn(HV, dtype=torch.bfloat16, device=device),
        "a": torch.randn(total, HV, dtype=torch.bfloat16, device=device),
        "b": torch.randn(total, HV, dtype=torch.bfloat16, device=device),
        "q": torch.randn(1, total, H, K, dtype=torch.bfloat16, device=device),
        "k": torch.randn(1, total, H, K, dtype=torch.bfloat16, device=device),
        "v": torch.randn(1, total, HV, V, dtype=torch.bfloat16, device=device),
        "state": torch.randn(B, HV, K, V, dtype=torch.float32, device=device),
        "indices": torch.arange(B, dtype=torch.int32, device=device),
        "cu_seqlens": torch.arange(
            0, B * tokens_per_request + 1, tokens_per_request, dtype=torch.int32, device=device),
    }
    if empty_last:
        inputs["cu_seqlens"][-1] = total
    if tokens_per_request > 1:
        # Target verify keeps the pool untouched and publishes the per-draft-token
        # states into a side buffer instead.
        inputs["buf"] = torch.zeros(
            B * tokens_per_request, HV, V, K, dtype=torch.float32, device=device)
        inputs["inter_idx"] = torch.arange(B, dtype=torch.int32, device=device)
    return inputs


def _call(inputs, tokens_per_request):
    verify = tokens_per_request > 1
    return k_last(
        A_log=inputs["A_log"], a=inputs["a"], dt_bias=inputs["dt_bias"],
        softplus_beta=1.0, softplus_threshold=20.0,
        q=inputs["q"], k=inputs["k"], v=inputs["v"], b=inputs["b"],
        initial_state_source=inputs["state"],
        initial_state_indices=inputs["indices"],
        scale=K ** -0.5, use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs["cu_seqlens"], is_kda=False,
        disable_state_update=verify,
        intermediate_states_buffer=inputs.get("buf"),
        intermediate_state_indices=inputs.get("inter_idx"),
        cache_steps_or_num_accept=tokens_per_request if verify else None,
        retrieve_parent_token=None,
    )


def _warmup(inputs, tokens_per_request):
    """Capture requires the op to have run on a side stream beforehand."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _call(inputs, tokens_per_request)
    torch.cuda.current_stream().wait_stream(side)


def _capture(inputs, tokens_per_request):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(inputs, tokens_per_request)
    return graph, out


def _capture_and_replay(inputs, tokens_per_request):
    """Run the op eagerly, then through a captured graph, and compare.

    Capture itself only records, so the state pool is reset before the replay to
    put it back in the state the eager run started from. Matching outputs then
    say the dump neither broke capture nor perturbed the kernel.
    """
    state, state0 = inputs["state"], inputs["state"].clone()

    _warmup(inputs, tokens_per_request)

    state.copy_(state0)
    out_eager = _call(inputs, tokens_per_request).clone()
    state_eager = state.clone()

    graph, out_graph = _capture(inputs, tokens_per_request)

    state.copy_(state0)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(out_graph, out_eager, atol=1e-3, rtol=1e-2)
    if not tokens_per_request > 1:
        torch.testing.assert_close(state, state_eager, atol=1e-3, rtol=1e-2)


def _phase_records(text):
    """Split captured stdout into {phase: [dump records]} on the @@PHASE marks."""
    phases, current = {}, None
    for line in text.splitlines():
        if line.startswith("@@PHASE "):
            current = line[len("@@PHASE "):].strip()
            phases[current] = []
        elif current is not None and line.startswith(("[GDN]", "[KDA]")):
            phases[current].append(line.strip())
    return phases


def _check_show_params_records(empty_last=False):
    """The dump keeps working under capture, minus the unreadable offsets.

    Redirect fd 1 rather than sys.stdout: the records are printed by C++, so a
    Python-level redirect would not see them. The @@PHASE marks go through the
    same fd, which is what keeps them ordered against the C++ output (the C++
    side ends every record with std::endl, so nothing lingers in a buffer across
    a phase boundary).
    """
    inputs = _make_inputs(DRAFT_TOKENS, empty_last=empty_last)
    # Warm up outside the redirect: only one eager call and the capture itself
    # may land in the phases, otherwise the buckets mix records.
    _warmup(inputs, DRAFT_TOKENS)
    with tempfile.TemporaryFile() as sink:
        saved = os.dup(1)
        try:
            sys.stdout.flush()
            os.dup2(sink.fileno(), 1)
            os.write(1, b"@@PHASE eager\n")
            _call(inputs, DRAFT_TOKENS)
            torch.cuda.synchronize()
            os.write(1, b"@@PHASE capture\n")
            _capture(inputs, DRAFT_TOKENS)
            os.write(1, b"@@PHASE end\n")
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        sink.seek(0)
        phases = _phase_records(sink.read().decode(errors="replace"))

    eager, capture = phases.get("eager", []), phases.get("capture", [])
    assert len(eager) == 1, f"expected one record for the eager call, got {eager}"
    assert len(capture) == 1, f"expected one record during capture, got {capture}"
    # A packed verify step is a [P] case with explicit offsets when they can be
    # read, and degrades to the shapes-only [D] form when they cannot.
    assert "cu_seqlens_q:" in eager[0], f"eager record lost its offsets: {eager[0]}"
    assert "cu_seqlens_q:" not in capture[0], \
        f"record emitted during capture claims offsets it cannot have read: {capture[0]}"
    if empty_last:
        assert "cu_seqlens_q:[0,4,4]" in eager[0], \
            f"empty request offsets were lost or changed: {eager[0]}"
    print(f"  eager record:   {eager[0]}")
    print(f"  capture record: {capture[0]}")


def run_worker(name):
    _load_device_dependencies()
    print(f"[{name}] PPU_LIB_PERF_INSTRUMENT="
          f"{os.environ.get('PPU_LIB_PERF_INSTRUMENT')} "
          f"PPU_LIB_SHOW_PARAMS={os.environ.get('PPU_LIB_SHOW_PARAMS')}")
    for tokens_per_request, label in ((1, "decode"), (DRAFT_TOKENS, "mtp verify")):
        for empty_last in (False, True):
            inputs = _make_inputs(tokens_per_request, empty_last=empty_last)
            empty_state = inputs["state"][-1].clone() if empty_last else None
            _capture_and_replay(inputs, tokens_per_request)
            if empty_last:
                torch.testing.assert_close(inputs["state"][-1], empty_state, atol=0, rtol=0)
            print(f"  [{name}] {label} empty_last={empty_last}: capture + replay matches eager")
    if name in ("show", "sanity_show"):
        _check_show_params_records()
        _check_show_params_records(empty_last=True)
    print(f"  [{name}] OK")


def run_driver():
    _load_device_dependencies()
    if not torch.cuda.is_available():
        print("SKIP: no device visible")
        return 0

    failures = []
    for name, env, why in CONFIGS:
        print(f"\n=== {name}: {why} ===")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", name],
            env={**os.environ, **env},
            capture_output=True, text=True,
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode == 0:
            print(f"--- {name} pass ---")
            continue
        failures.append(name)
        sys.stdout.write(proc.stderr)
        print(f"--- {name} FAIL (exit {proc.returncode}) ---")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"FAILED configs: {', '.join(failures)}")
        return 1
    print(f"All {len(CONFIGS)} instrumentation configs survive graph capture")
    return 0


class OpForgeCaptureContractTest(unittest.TestCase):
    def test_error_and_capture_never_copy_device_values(self):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("requires a C++17 host compiler")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="pla-capture-contract-") as directory:
            tmp = Path(directory)
            (tmp / "ATen/cuda").mkdir(parents=True)
            (tmp / "c10/cuda").mkdir(parents=True)
            (tmp / "ATen/ATen.h").write_text(r'''
#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("contract"); } while (0)
#define TORCH_WARN_ONCE(...) do {} while (0)
inline int host_copies = 0;
namespace at {
enum ScalarType { BFloat16, Half, Float, Int, Long };
constexpr auto kLong = ScalarType::Long;
constexpr auto kInt = ScalarType::Int;
constexpr int kCPU = 0;
struct TensorOptions {
    TensorOptions device(int) const { return {}; }
    TensorOptions dtype(ScalarType) const { return {}; }
    TensorOptions pinned_memory(bool) const { return {}; }
};
struct IntList {
    std::vector<int64_t> data;
    friend std::ostream& operator<<(std::ostream& os, const IntList& list) {
        os << "[";
        for (std::size_t i = 0; i < list.data.size(); ++i) { if (i) os << ","; os << list.data[i]; }
        return os << "]";
    }
};
struct Tensor {
    std::vector<int64_t> values;
    Tensor() = default;
    Tensor(std::initializer_list<int64_t> v) : values(v) {}
    bool defined() const { return true; }
    int dim() const { return 1; }
    bool is_cuda() const { return false; }
    int device() const { return 0; }
    int get_device() const { return 0; }
    int64_t stride(int) const { return 1; }
    int64_t element_size() const { return 8; }
    int64_t numel() const { return values.size(); }
    ScalarType scalar_type() const { return Long; }
    TensorOptions options() const { return {}; }
    IntList sizes() const { return {values}; }
    IntList strides() const { return {{1}}; }
    Tensor to(TensorOptions) const { ++host_copies; return *this; }
    Tensor to(ScalarType) const { return *this; }
    Tensor contiguous() const { return *this; }
    template<class T> T* data_ptr() const {
        return reinterpret_cast<T*>(const_cast<int64_t*>(values.data()));
    }
    void* data_ptr() const { return const_cast<int64_t*>(values.data()); }
};
inline Tensor empty(std::initializer_list<int64_t>, TensorOptions) { return {}; }
}
''')
            (tmp / "ATen/cuda/CUDAContext.h").write_text(r'''
#pragma once
namespace at { namespace cuda {
struct Stream { int stream() const { return 0; } };
inline Stream getCurrentCUDAStream() { return {}; }
inline Stream getCurrentCUDAStream(int) { return {}; }
}}
''')
            (tmp / "cuda_runtime_api.h").write_text(r'''
#pragma once
#include <cstddef>
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated
};
typedef int cudaError_t;
constexpr int cudaSuccess = 0, cudaErrorUnknown = 999;
enum cudaMemcpyKind { cudaMemcpyDeviceToHost };
inline int probe_result = cudaSuccess, cleared_errors = 0;
inline auto probe_status = cudaStreamCaptureStatusNone;
inline int cudaStreamIsCapturing(int, cudaStreamCaptureStatus* out) {
    *out = probe_status;
    return probe_result;
}
inline int cudaGetLastError() { ++cleared_errors; return probe_result; }
inline const char* cudaGetErrorString(cudaError_t) { return "mock"; }
inline int cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, int) { return cudaSuccess; }
inline int cudaStreamSynchronize(int) { return cudaSuccess; }
''')
            (tmp / "c10/cuda/CUDAGuard.h").write_text(r'''
#pragma once
#include <exception>
namespace c10 {
struct Error : std::exception {
    const char* what() const noexcept override { return "c10"; }
    const char* what_without_backtrace() const { return "c10"; }
};
namespace cuda { struct CUDAGuard { explicit CUDAGuard(int) {} }; }
}
''')
            source = tmp / "contract.cpp"
            source.write_text(r'''
#include "src/op_forge_case.h"
#include <cassert>

int main() {
    const at::Tensor offsets{0, 2, 4};
    const auto emit = [&]() {
        return generate_nvtx_tag(2, 2, 128, 128, 4, 8, "bf16", false, &offsets);
    };
    assert(emit().find("cu_seqlens_q:[0,2,4]") != std::string::npos);
    assert(host_copies == 1);
    for (auto state : {cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated}) {
        host_copies = 0;
        probe_status = state;
        assert(emit().find("cu_seqlens_q:") == std::string::npos);
        assert(host_copies == 0);
    }
    host_copies = 0;
    probe_status = cudaStreamCaptureStatusNone;
    probe_result = cudaErrorUnknown;
    assert(emit().find("cu_seqlens_q:") == std::string::npos);
    assert(host_copies == 0 && cleared_errors == 1);

    probe_result = cudaSuccess;
    const at::Tensor empty_request{0, 2, 2};
    auto tag = generate_nvtx_tag(2, 2, 128, 128, 4, 8, "bf16", false, &empty_request);
    assert(tag.find("cu_seqlens_q:[0,2,2]") != std::string::npos);
    const at::Tensor backwards{0, 2, 1};
    bool rejected = false;
    try {
        generate_nvtx_tag(2, 2, 128, 128, 4, 8, "bf16", false, &backwards);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
''')
            binary = tmp / "contract"
            subprocess.run([compiler, "-std=c++17", "-I", str(tmp), "-I", str(root),
                            str(source), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", choices=[name for name, _, _ in CONFIGS],
                        help="run one configuration in this process "
                             "(the env vars must already be set)")
    parser.add_argument("--cpu-only", action="store_true", help="run only the serializer's CPU contract tests")
    args = parser.parse_args()
    if args.cpu_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(OpForgeCaptureContractTest)
        return int(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    if args.worker:
        run_worker(args.worker)
        return 0
    return run_driver()


if __name__ == "__main__":
    raise SystemExit(main())
