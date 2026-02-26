#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

class GroupNorm {
public:
    using schema = void (*)(Tensor, int64_t, std::optional<Tensor>, std::optional<Tensor>, double, Tensor);
    static void execute(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps, Tensor output);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor group_norm(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps);
void group_norm_(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps, Tensor output);

} // namespace infinicore::op