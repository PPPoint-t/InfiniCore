#include "infinicore/ops/isclose.hpp"

namespace infinicore::op {

common::OpDispatcher<IsClose::schema> &IsClose::dispatcher() {
    static common::OpDispatcher<IsClose::schema> dispatcher_;
    return dispatcher_;
}

void IsClose::execute(Tensor a, Tensor b, double rtol, double atol, bool equal_nan, Tensor out) {
    infinicore::context::setDevice(a->device());
    dispatcher().lookup(a->device().getType())(a, b, rtol, atol, equal_nan, out);
}

Tensor isclose(Tensor a, Tensor b, double rtol, double atol, bool equal_nan) {
    auto out = Tensor::empty(a->shape(), DataType::BOOL, a->device());
    IsClose::execute(a, b, rtol, atol, equal_nan, out);
    return out;
}

} // namespace infinicore::op