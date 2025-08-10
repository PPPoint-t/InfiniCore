#include "cast_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include "infinicore.h"
#include <algorithm>

namespace op::cast::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info_r = CastInfo::create(out_desc, in_desc);
    CHECK_RESULT(info_r);

    *desc_ptr = new Descriptor(
        info_r.take(),
        0,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

template <typename Tout, typename Tin>
static inline void cpu_cast_impl_incremental(
    void *output, const void *input, const op::cast::CastInfo &info) {

    const size_t ndim = info.shape.size();
    const size_t n = info.n;

    auto out_base = reinterpret_cast<Tout *>(output);
    auto in_base = reinterpret_cast<const Tin *>(input);

    const std::vector<size_t> &shape = info.shape;
    const std::vector<ptrdiff_t> &in_stride = info.in_stride;
    const std::vector<ptrdiff_t> &out_stride = info.out_stride;

    if (n == 0) return;

    std::vector<size_t> idx(ndim, 0);
    ptrdiff_t in_off = 0;
    ptrdiff_t out_off = 0;

    for (size_t it = 0; it < n; ++it) {
        const Tin *in_elem = in_base + in_off;
        Tout *out_elem = out_base + out_off;
        *out_elem = utils::cast<Tout, Tin>(*in_elem);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            idx[d] += 1;
            if (in_stride[d] != 0) in_off += in_stride[d];
            if (out_stride[d] != 0) out_off += out_stride[d];

            if (idx[d] < shape[d]) {
                break;
            } else {
                idx[d] = 0;
                if (in_stride[d] != 0) in_off -= static_cast<ptrdiff_t>(shape[d]) * in_stride[d];
                if (out_stride[d] != 0) out_off -= static_cast<ptrdiff_t>(shape[d]) * out_stride[d];
            }
        }
    }
}

// 修改 Algo，新增 info 参数并调用新的实现
struct Algo {
    template <class Tout, class Tin>
    infiniStatus_t run(
        void *workspace, size_t workspace_size,
        void *output, const void *input, size_t n,
        const op::cast::CastInfo &info, void *stream) const {

        (void)workspace; (void)workspace_size; (void)stream;
        cpu_cast_impl_incremental<Tout, Tin>(output, input, info);
        return INFINI_STATUS_SUCCESS;
    }
};

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    if (output == const_cast<void*>(input)) {
        return INFINI_STATUS_BAD_PARAM; // or INFINI_STATUS_INPLACE_NOT_SUPPORTED
    }

    Calculate::calculate<Algo>(
        Algo{}, _info, workspace, workspace_size,
        output, input, stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cast::cpu
