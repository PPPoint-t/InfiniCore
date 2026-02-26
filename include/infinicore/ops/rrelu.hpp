#pragma once
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class RReLU {
public:
    using schema = void (*)(Tensor, double, double, bool, Tensor);
    static void execute(Tensor input, double lower, double upper, bool training, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor rrelu(Tensor input, double lower = 0.125, double upper = 0.3333333333333333, bool training = false, bool inplace = false);
// In-place
Tensor& rrelu_(Tensor input, double lower = 0.125, double upper = 0.3333333333333333, bool training = false);

} // namespace infinicore::op