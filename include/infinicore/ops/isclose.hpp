#pragma once
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class IsClose {
public:
    using schema = void (*)(Tensor, Tensor, double, double, bool, Tensor);
    static void execute(Tensor a, Tensor b, double rtol, double atol, bool equal_nan, Tensor out);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor isclose(Tensor a, Tensor b, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);

} // namespace infinicore::op