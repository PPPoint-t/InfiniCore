#include "infinicore/ops/rrelu.hpp"

namespace infinicore::op {

common::OpDispatcher<RReLU::schema> &RReLU::dispatcher() {
    static common::OpDispatcher<RReLU::schema> dispatcher_;
    return dispatcher_;
}

void RReLU::execute(Tensor input, double lower, double upper, bool training, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, lower, upper, training, output);
}

Tensor rrelu(Tensor input, double lower, double upper, bool training, bool inplace) {
    if (inplace) {
        RReLU::execute(input, lower, upper, training, input);
        return input;
    } else {
        auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
        RReLU::execute(input, lower, upper, training, out);
        return out;
    }
}

Tensor& rrelu_(Tensor input, double lower, double upper, bool training) {
    RReLU::execute(input, lower, upper, training, input);
    return input;
}

} // namespace infinicore::op