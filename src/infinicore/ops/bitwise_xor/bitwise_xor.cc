#include "infinicore/ops/bitwise_xor.hpp"

namespace infinicore::op {

common::OpDispatcher<BitwiseXor::schema> &BitwiseXor::dispatcher() {
    static common::OpDispatcher<BitwiseXor::schema> dispatcher_;
    return dispatcher_;
}

void BitwiseXor::execute(Tensor a, Tensor b, Tensor out) {
    infinicore::context::setDevice(a->device());
    dispatcher().lookup(a->device().getType())(a, b, out);
}

Tensor bitwise_xor(Tensor a, Tensor b) {
    auto out = Tensor::empty(a->shape(), a->dtype(), a->device());
    bitwise_xor_out(a, b, out);
    return out;
}

Tensor& bitwise_xor_out(Tensor a, Tensor b, Tensor& out) {
    BitwiseXor::execute(a, b, out);
    return out;
}

} // namespace infinicore::op