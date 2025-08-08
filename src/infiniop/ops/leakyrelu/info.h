#ifndef __LEAKYRELU_INFO_H__
#define __LEAKYRELU_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::leakyrelu {

class LeakyreluInfo {
    LeakyreluInfo() = default;

public:
    infiniDtype_t atype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> input_strides;
    float negative_slope;

    size_t ndim() const { return shape.size(); }
    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    static utils::Result<LeakyreluInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        float negative_slope) 
    {
        if (!output_desc || !input_desc) return INFINI_STATUS_BAD_PARAM;
        if (output_desc->dtype() != input_desc->dtype()) return INFINI_STATUS_BAD_TENSOR_DTYPE;

        auto dt = output_desc->dtype();
        if (dt != INFINI_DTYPE_F16 && dt != INFINI_DTYPE_BF16 &&
            dt != INFINI_DTYPE_F32 && dt != INFINI_DTYPE_F64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (output_desc->ndim() != input_desc->ndim()) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        size_t ndim = output_desc->ndim();
        const auto out_shape = output_desc->shape();
        const auto in_shape = input_desc->shape();
        for (size_t i = 0; i < ndim; ++i) {
            if (out_shape[i] != in_shape[i]) return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<LeakyreluInfo>(LeakyreluInfo{
            dt,
            output_desc->shape(),
            output_desc->strides(),
            input_desc->strides(),
            negative_slope
        });
    }
};

} // namespace op::leakyrelu

#endif // __LEAKYRELU_INFO_H__
