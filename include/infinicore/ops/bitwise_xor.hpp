#pragma once
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class BitwiseXor {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor a, Tensor b, Tensor out);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor bitwise_xor(Tensor a, Tensor b);
Tensor& bitwise_xor_out(Tensor a, Tensor b, Tensor& out);

} // namespace infinicore::op