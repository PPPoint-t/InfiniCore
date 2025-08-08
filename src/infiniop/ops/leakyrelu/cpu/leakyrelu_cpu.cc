#include "leakyrelu_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
namespace op::leakyrelu::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    float negative_slope) {

    auto result = LeakyreluInfo::create(output_desc, input_desc, negative_slope);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(nullptr, std::move(info), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// Helper: check if tensor is C-contiguous (row-major) based on shape & strides (element strides)
static inline bool is_contiguous(const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides) {
    size_t ndim = shape.size();
    if (ndim == 0) return true;
    // last stride must be 1
    if (strides[ndim - 1] != 1) return false;
    size_t expected = 1;
    // verify stride[d] == product(shape[d+1..end])
    for (int d = int(ndim) - 1; d >= 0; --d) {
        if (strides[d] != ptrdiff_t(expected)) return false;
        expected *= shape[d];
    }
    return true;
}

// Helper: compute offset (in elements) from linear idx for arbitrary strides/layout
static inline size_t calc_offset_from_index(size_t idx, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides) {
    size_t ndim = shape.size();
    size_t t = idx;
    ptrdiff_t off = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) {
        size_t coord = t % shape[d];
        t /= shape[d];
        off += ptrdiff_t(coord) * strides[d];
    }
    return static_cast<size_t>(off);
}

// contiguous fast path (works when in/out are contiguous)
template <typename T>
infiniStatus_t leakyrelu_contiguous(const LeakyreluInfo *info, T *out, const T *in) {
    size_t n = info->numel();
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(n); ++i) {
        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            float xf = utils::cast<float>(in[i]);
            float yf = xf >= 0.f ? xf : xf * info->negative_slope;
            out[i] = utils::cast<T>(yf);
        } else if constexpr (std::is_floating_point<T>::value) {
            T xin = in[i];
            out[i] = xin >= (T)0 ? xin : static_cast<T>(static_cast<double>(xin) * info->negative_slope);
        } else {
            // integral types: keep identity (or you can change to 0 if desired)
            out[i] = in[i];
        }
    }
    return INFINI_STATUS_SUCCESS;
}

// generic path (supports arbitrary strides / non-contiguous layout)
template <typename T>
infiniStatus_t leakyrelu_generic(const LeakyreluInfo *info, T *out, const T *in) {
    size_t n = info->numel();
    const auto &shape = info->shape;
    const auto &istr = info->input_strides;
    const auto &ostr = info->output_strides;

#pragma omp parallel for
    for (ptrdiff_t idx = 0; idx < ptrdiff_t(n); ++idx) {
        size_t in_off = calc_offset_from_index(static_cast<size_t>(idx), shape, istr);
        size_t out_off = calc_offset_from_index(static_cast<size_t>(idx), shape, ostr);

        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            float xf = utils::cast<float>(*(const T *)((const char *)in + in_off * sizeof(T)));
            float yf = xf >= 0.f ? xf : xf * info->negative_slope;
            *(T *)((char *)out + out_off * sizeof(T)) = utils::cast<T>(yf);
        } else if constexpr (std::is_floating_point<T>::value) {
            T xin = *(const T *)((const char *)in + in_off * sizeof(T));
            T outv = xin >= (T)0 ? xin : static_cast<T>(static_cast<double>(xin) * info->negative_slope);
            *(T *)((char *)out + out_off * sizeof(T)) = outv;
        } else {
            // integral types
            T v = *(const T *)((const char *)in + in_off * sizeof(T));
            *(T *)((char *)out + out_off * sizeof(T)) = v;
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output, const void *input,
    void *stream) const {

    (void)workspace; (void)workspace_size; (void)stream;

    // decide contiguous vs generic based on shape & strides
    bool in_contig = is_contiguous(_info.shape, _info.input_strides);
    bool out_contig = is_contiguous(_info.shape, _info.output_strides);

    if (_info.atype == INFINI_DTYPE_F16) {
        if (in_contig && out_contig) {
            CHECK_STATUS(leakyrelu_contiguous<fp16_t>(&_info, (fp16_t *)output, (const fp16_t *)input));
        } else {
            CHECK_STATUS(leakyrelu_generic<fp16_t>(&_info, (fp16_t *)output, (const fp16_t *)input));
        }
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        if (in_contig && out_contig) {
            CHECK_STATUS(leakyrelu_contiguous<bf16_t>(&_info, (bf16_t *)output, (const bf16_t *)input));
        } else {
            CHECK_STATUS(leakyrelu_generic<bf16_t>(&_info, (bf16_t *)output, (const bf16_t *)input)); // fixed below
        }
    } else if (_info.atype == INFINI_DTYPE_F32) {
        if (in_contig && out_contig) {
            CHECK_STATUS(leakyrelu_contiguous<float>(&_info, (float *)output, (const float *)input));
        } else {
            CHECK_STATUS(leakyrelu_generic<float>(&_info, (float *)output, (const float *)input));
        }
    } else if (_info.atype == INFINI_DTYPE_F64) {
        if (in_contig && out_contig) {
            CHECK_STATUS(leakyrelu_contiguous<double>(&_info, (double *)output, (const double *)input));
        } else {
            CHECK_STATUS(leakyrelu_generic<double>(&_info, (double *)output, (const double *)input));
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::leakyrelu::cpu
