#pragma once
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// Reshape 主要是元数据操作，不需要像 Compute Op 那样定义 Schema/Dispatcher
Tensor reshape(Tensor input, Shape shape);

} // namespace infinicore::op