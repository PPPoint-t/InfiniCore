#pragma once
#include "infinicore/ops/rrelu.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_rrelu(py::module &m) {
    m.def("rrelu", &op::rrelu,
          py::arg("input"),
          py::arg("lower") = 0.125,
          py::arg("upper") = 0.3333333333333333,
          py::arg("training") = false,
          py::arg("inplace") = false,
          "Applies the randomized leaky rectified linear unit function.");
}

} // namespace infinicore::ops