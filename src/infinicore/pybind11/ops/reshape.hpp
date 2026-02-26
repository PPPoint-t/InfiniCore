#pragma once
#include "infinicore/ops/reshape.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_reshape(py::module &m) {
    m.def("reshape",
          &op::reshape,
          py::arg("input"),
          py::arg("shape"),
          R"doc(Returns a tensor with the same data and number of elements as input, but with the specified shape.)doc");
}

} // namespace infinicore::ops