/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <torch/extension.h>
#include "runtime.h"
#include "../../../../bypass_exec.h"
#include "../../../../op_forge_case.h"
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
std::optional<std::vector<int64_t>> raw_values(const py::object& raw) {
  if (raw.is_none()) return std::nullopt;
  auto values = py::reinterpret_steal<py::tuple>(PySequence_Tuple(raw.ptr()));
  if (!values) throw py::error_already_set();
  if (values.size() < 2 || values.size() > 65536) throw py::value_error("illegal raw offsets length");
  std::vector<int64_t> result;
  result.reserve(values.size());
  for (const auto item : values) {
    if (PyBool_Check(item.ptr())) throw py::value_error("offset must not be bool");
    auto integer = py::reinterpret_steal<py::object>(PyNumber_Index(item.ptr()));
    if (!integer) { PyErr_Clear(); throw py::value_error("offset must be an integer"); }
    auto value = PyLong_AsLongLong(integer.ptr());
    if (PyErr_Occurred()) { PyErr_Clear(); throw py::value_error("offset exceeds the int64 range"); }
    result.push_back(value);
  }
  return result;
}

void forward(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor g, at::Tensor beta,
             at::Tensor a, at::Tensor bias, at::Tensor out, std::optional<double> scale,
             double lower_bound, std::optional<at::Tensor> initial, std::optional<at::Tensor> final,
             std::optional<at::Tensor> cu, std::optional<at::Tensor> pool,
             std::optional<at::Tensor> indices, py::object raw) {
  fla_runtime::Arguments args{q, k, v, g, beta, a, bias, out};
  if (scale) args.scale = *scale;
  args.lower_bound = lower_bound;
  args.initial = initial.value_or(at::Tensor()); args.final = final.value_or(at::Tensor());
  args.cu = cu.value_or(at::Tensor()); args.pool = pool.value_or(at::Tensor());
  args.indices = indices.value_or(at::Tensor()); args.raw = raw_values(raw);
  py::gil_scoped_release release;
  fla_runtime::forward(std::move(args));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  (void)pla::bypass::enabled();
  (void)pla::instrument::show_params();
  module.attr("runtime_abi") = fla_runtime::kRuntimeAbi;
  module.attr("kernel_abi") = fla_runtime::kKernelAbi;
  module.attr("cp_abi") = fla_runtime::kCpAbi;
  module.attr("cp_heads") = py::cast(fla_runtime::supported_cp_heads());
  module.attr("supported_heads") = py::cast(fla_runtime::supported_heads());
  module.attr("generic_head_max") = 65535;
  module.attr("manifest_schema") = "tc01_fla_monolithic_build_v2";
  module.def("forward", &forward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
             py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"), py::arg("out"),
             py::arg("scale") = py::none(), py::arg("lower_bound") = -5.,
             py::arg("initial_state") = py::none(), py::arg("final_state") = py::none(),
             py::arg("cu_seqlens") = py::none(), py::arg("state_pool") = py::none(),
             py::arg("state_indices") = py::none(), py::arg("raw_cu_seqlens") = py::none());
  module.def("_metadata_only", &fla_runtime::metadata_only, py::arg("input"),
             py::arg("tokens"), py::arg("heads"), py::call_guard<py::gil_scoped_release>());
}
