#include "infinicore/ops/group_norm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<GroupNorm::schema> &GroupNorm::dispatcher() {
    static common::OpDispatcher<GroupNorm::schema> dispatcher_;
    return dispatcher_;
};

void GroupNorm::execute(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, num_groups, weight, bias, eps, output);
}

Tensor group_norm(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    group_norm_(input, num_groups, weight, bias, eps, output);
    return output;
}

void group_norm_(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps, Tensor output) {
    GroupNorm::execute(input, num_groups, weight, bias, eps, output);
}

} // namespace infinicore::op