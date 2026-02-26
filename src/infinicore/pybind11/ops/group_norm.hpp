#pragma once

#include "infinicore/ops/group_norm.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_group_norm(py::module &m) {
    m.def("group_norm",
          &op::group_norm,
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight") = py::none(),
          py::arg("bias") = py::none(),
          py::arg("eps") = 1e-5,
          R"doc(Applies Group Normalization.)doc");

    m.def("group_norm_",
          &op::group_norm_,
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps"),
          py::arg("output"),
          R"doc(In-place Group Normalization.)doc");
}

} // namespace infinicore::ops