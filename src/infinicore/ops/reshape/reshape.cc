#include "infinicore/ops/reshape.hpp"
#include "infinicore/tensor.hpp"

namespace infinicore::op {

Tensor reshape(Tensor input, Shape shape) {
    try {
        return input->view(shape);
    } catch (...) {
        return input->contiguous()->view(shape);
    }
}

} // namespace infinicore::op