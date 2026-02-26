#pragma once
#include "infinicore/ops/isclose.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_isclose(py::module &m) {
    m.def("isclose", &op::isclose, 
          py::arg("input"), py::arg("other"), 
          py::arg("rtol") = 1e-05, 
          py::arg("atol") = 1e-08, 
          py::arg("equal_nan") = false,
          "Returns a boolean tensor where two tensors are element-wise equal within a tolerance.");
}

} // namespace infinicore::ops