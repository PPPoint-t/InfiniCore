#pragma once
#include "infinicore/ops/bitwise_xor.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_bitwise_xor(py::module &m) {
    m.def("bitwise_xor", &op::bitwise_xor, py::arg("input"), py::arg("other"),
          "Computes the bitwise XOR of input and other.");
          
    m.def("bitwise_xor_out", &op::bitwise_xor_out, py::arg("input"), py::arg("other"), py::arg("out"),
          "Computes the bitwise XOR of input and other, writing result to out.");
}

} // namespace infinicore::ops