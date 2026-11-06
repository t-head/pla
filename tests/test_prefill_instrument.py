# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU C++ contract tests for prefill UT tags, range lifetime and capture safety.

Run: python tests/test_prefill_instrument.py
The production serializer and range are compiled with small ATen/HGTX doubles;
this does not replace an SDK build or profiling on PPU.
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Only device/runtime dependencies are doubled; serialization and RAII are real.
PREAMBLE = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("check"); } while (0)
#define TORCH_WARN_ONCE(...) do {} while (0)
int copies = 0, queries = 0, depth = 0, pushes = 0, pops = 0;
int capture_mode = 0;
std::string last_tag;

using hggcStream_t = void*;
enum hggcStreamCaptureStatus { hggcStreamCaptureStatusNone, hggcStreamCaptureStatusActive };
constexpr int hggcSuccess = 0;
int hggcStreamIsCapturing(hggcStream_t, hggcStreamCaptureStatus* status) {
    ++queries;
    if (capture_mode == 2) return 1;
    *status = capture_mode == 1 ? hggcStreamCaptureStatusActive : hggcStreamCaptureStatusNone;
    return hggcSuccess;
}
int hggcGetLastError() { return 0; }

namespace at {
enum class ScalarType { BFloat16, Half, Float, Int, Long };
constexpr int kCPU = 0;
constexpr ScalarType kLong = ScalarType::Long;
struct TensorOptions {
    TensorOptions& device(int) { return *this; }
    TensorOptions& dtype(ScalarType) { return *this; }
};
struct Tensor {
    std::vector<int64_t> values;
    ScalarType type = ScalarType::Int;
    bool gpu = true;
    bool defined() const { return true; }
    int dim() const { return 1; }
    ScalarType scalar_type() const { return type; }
    Tensor to(const TensorOptions&) const {
        if (gpu) { ++copies; assert(capture_mode == 0); }
        Tensor result = *this;
        result.gpu = false;
        return result;
    }
    Tensor contiguous() const { return *this; }
    template<class T> const T* data_ptr() const { return reinterpret_cast<const T*>(values.data()); }
    size_t numel() const { return values.size(); }
};
namespace cuda {
struct Stream { void* stream() const { return nullptr; } };
Stream getCurrentCUDAStream() { return {}; }
}
}
namespace hgtx3 {
class scoped_range {
public:
    explicit scoped_range(const char* message) : message_(message) {
        last_tag = message;
        ++depth;
        ++pushes;
    }
    ~scoped_range() {
        // Verify the caller-owned message has not expired before pop.
        assert(last_tag == message_);
        --depth;
        ++pops;
    }
    scoped_range(const scoped_range&) = delete;
    scoped_range& operator=(const scoped_range&) = delete;
private:
    const char* message_;
};
}
'''

MAIN = r'''
int main(int argc, char** argv) {
    assert(argc == 5);
    capture_mode = std::atoi(argv[1]);
    const bool is_kda = std::atoi(argv[2]);
    const bool fixed = std::atoi(argv[3]);
    const bool unwind = std::atoi(argv[4]);
    // Match the import-time snapshot in each extension binding.
    const bool perf = pla::instrument::perf_instrument();
    const bool show = pla::instrument::show_params();
    const bool enabled = perf || show;
    setenv("PPU_LIB_PERF_INSTRUMENT", perf ? "0" : "1", 1);
    setenv("PPU_LIB_SHOW_PARAMS", show ? "0" : "1", 1);
    at::Tensor cu{{0, 17, 17, 49}};
    const int batch = fixed ? 1 : 3;
    const int heads = is_kda ? 12 : 2;
    std::ostringstream sink;
    auto* saved = std::cout.rdbuf(sink.rdbuf());
    try {
        const pla::instrument::PrefillRange range(
            batch, 64, 128, 128, heads, 12, "bf16", is_kda,
            fixed ? nullptr : &cu, -3.0f);
        assert(depth == int(perf));
        if (perf) {
            assert(last_tag.find(is_kda ? "[KDA]:[P][" : "[GDN]:[P][") == 0);
            assert(last_tag.find("num_heads_qk:" + std::to_string(heads)) != std::string::npos);
            assert(last_tag.find("num_heads_v:12") != std::string::npos);
            assert(last_tag.find("dtype:bf16") != std::string::npos);
            assert((last_tag.find("gate_lower_bound:-3.0") != std::string::npos) == is_kda);
            if (fixed) assert(last_tag.find("cu_seqlens_q:[0,64]") != std::string::npos);
            else if (capture_mode == 0)
                assert(last_tag.find("cu_seqlens_q:[0,17,17,49]") != std::string::npos);
            else assert(last_tag.find("cu_seqlens_q:") == std::string::npos);
        }
        if (show && perf) assert(sink.str() == last_tag + "\n");
        if (unwind) throw std::runtime_error("simulated launch failure");
    } catch (const std::runtime_error&) {
        assert(unwind);
    }
    std::cout.rdbuf(saved);
    assert(depth == 0 && pushes == int(perf) && pops == int(perf));
    assert(copies == int(enabled && !fixed && capture_mode == 0));
    assert(queries == int(enabled && !fixed));
    assert(sink.str().empty() == !show);
    if (show) {
        assert(sink.str().find(is_kda ? "[KDA]:[P][" : "[GDN]:[P][") == 0);
        assert((sink.str().find("cu_seqlens_q:") != std::string::npos) ==
               (fixed || capture_mode == 0));
    }
}
'''


class PrefillInstrumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++")
        if not compiler:
            raise unittest.SkipTest("requires a C++17 compiler")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        case_source = (ROOT / "src/op_forge_case.h").read_text()
        # Compile the real serializer and show-params snapshot without the
        # unrelated bypass Tensor helpers that need the full ATen runtime.
        serializer = case_source.split("static inline std::string get_dtype_string", 1)[1]
        serializer = "static inline std::string get_dtype_string" + serializer.split("namespace pla {", 1)[0]
        show = case_source.split('inline __attribute__((visibility("hidden"))) int show_params()', 1)[1]
        show = 'inline __attribute__((visibility("hidden"))) int show_params()' + show.split("inline void check_cuda", 1)[0]
        range_source = (ROOT / "src/prefill/instrument.h").read_text()
        range_source = "\n".join(line for line in range_source.splitlines()
                                 if not line.startswith(("#include", "#pragma")))
        source = PREAMBLE + serializer + "namespace pla { namespace instrument {\n" + show + "}}\n"
        source += range_source + MAIN
        (root / "contract.cc").write_text(source)
        cls.binary = root / "contract"
        result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra",
                                 str(root / "contract.cc"), "-o", str(cls.binary)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_compiled_range_and_capture_contract(self):
        for perf, show in ((0, 0), (1, 0), (0, 1), (1, 1)):
            env = dict(os.environ, PPU_LIB_PERF_INSTRUMENT=str(perf),
                       PPU_LIB_SHOW_PARAMS=str(show))
            # Eager, active capture and unknown capture status must all work.
            for capture in (0, 1, 2):
                for kda in (0, 1):
                    for fixed in (0, 1):
                        for unwind in (0, 1):
                            with self.subTest(perf=perf, show=show, capture=capture,
                                              kda=kda, fixed=fixed, unwind=unwind):
                                result = subprocess.run(
                                    [str(self.binary), str(capture), str(kda), str(fixed), str(unwind)],
                                    env=env, capture_output=True, text=True, timeout=10)
                                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtime_record_parser(self):
        path = ROOT / "tests/test_prefill_instrument_graph_capture.py"
        spec = importlib.util.spec_from_file_location("prefill_graph_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for family in ("gdn", "gdn-packed", "kda-legacy", "kda-fla"):
            prefix = "[KDA]:[P][3,200]" if family.startswith("kda") else "[GDN]:[P][3,200]"
            tag = prefix + "op:head_dim_qk:128,dtype:bf16"
            text = (f"@@PHASE eager\n{tag},cu_seqlens_q:[0,17,17,65].\n"
                    f"@@PHASE capture\n{tag}.\n@@PHASE replay\n@@PHASE end\n")
            module.check_records(text, family, True)
            module.check_records("@@PHASE eager\n@@PHASE capture\n@@PHASE replay\n@@PHASE end\n",
                                 family, False)
            for invalid in (text.replace(f"{tag}.", f"{tag},cu_seqlens_q:[0,72]."),
                            text.replace("@@PHASE replay\n", f"@@PHASE replay\n{tag}.\n"),
                            text.replace("@@PHASE end\n", "")):
                with self.assertRaises(AssertionError):
                    module.check_records(invalid, family, True)

    def test_all_prefill_launchers_keep_the_range_in_launch_scope(self):
        paths = {
            "src/prefill/flashqla/host/torch_extension_cute.cc": "launch_fused_fwd_cp_cute(",
            "src/prefill/flashkdapro/host/torch_extension_cute.cc": "launch_flashkda_prepare(",
            "src/prefill/flashkdapro/tc01/fla/runtime.cc": "initialize(a.q.get_device()",
        }
        for path, launch in paths.items():
            with self.subTest(path=path):
                source = (ROOT / path).read_text()
                after = source.split("const pla::instrument::PrefillRange profile_range(", 1)[1]
                self.assertIn(launch, after)
                self.assertNotIn("push_hgtx_and_print", after)
        for path in ("src/prefill/flashqla/host/torch_extension_cute.cc",
                     "src/prefill/flashkdapro/host/torch_extension_cute.cc",
                     "src/prefill/flashkdapro/tc01/fla/runtime_binding.cc"):
            self.assertIn("(void)pla::instrument::perf_instrument();", (ROOT / path).read_text())


if __name__ == "__main__":
    unittest.main()
