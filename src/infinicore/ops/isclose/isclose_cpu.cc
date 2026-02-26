#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/isclose.hpp"
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>

namespace infinicore::op::isclose_impl::cpu {

inline float bf16_to_f32(uint16_t val) {
    union { uint32_t i; float f; } u;
    u.i = static_cast<uint32_t>(val) << 16;
    return u.f;
}

template <typename T>
inline double get_val(const T* ptr, int64_t idx) {
    return utils::cast<double>(ptr[idx]);
}

template <>
inline double get_val<uint16_t>(const uint16_t* ptr, int64_t idx) {
    return static_cast<double>(bf16_to_f32(ptr[idx]));
}

inline bool check_isclose(double a, double b, double rtol, double atol, bool equal_nan) {
    if (std::isnan(a) || std::isnan(b)) {
        return equal_nan && std::isnan(a) && std::isnan(b);
    }
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

template <typename T>
void isclose_kernel(const Tensor& a, const Tensor& b, double rtol, double atol, bool equal_nan, Tensor& out) {
    int64_t numel = out->numel();
    if (numel == 0) return;

    const T* a_data = reinterpret_cast<const T*>(a->data());
    const T* b_data = reinterpret_cast<const T*>(b->data());
    bool* out_data = reinterpret_cast<bool*>(out->data());

    if (a->is_contiguous() && b->is_contiguous() && out->is_contiguous()) {
        #pragma omp parallel for if(numel > 4096)
        for (int64_t i = 0; i < numel; ++i) {
            double val_a = get_val<T>(a_data, i);
            double val_b = get_val<T>(b_data, i);
            out_data[i] = check_isclose(val_a, val_b, rtol, atol, equal_nan);
        }
        return;
    }

    int ndim = out->ndim();
    if (ndim == 0) {
        out_data[0] = check_isclose(get_val<T>(a_data, 0), get_val<T>(b_data, 0), rtol, atol, equal_nan);
        return;
    }

    auto out_shape = out->shape();
    auto out_strides = out->strides();
    auto a_shape = a->shape();
    auto a_strides = a->strides();
    auto b_shape = b->shape();
    auto b_strides = b->strides();

    std::vector<int64_t> a_broadcast_strides(ndim);
    std::vector<int64_t> b_broadcast_strides(ndim);
    int a_offset_dim = ndim - a->ndim();
    int b_offset_dim = ndim - b->ndim();

    for (int i = 0; i < ndim; ++i) {
        if (i < a_offset_dim) a_broadcast_strides[i] = 0;
        else a_broadcast_strides[i] = (a_shape[i - a_offset_dim] == 1) ? 0 : a_strides[i - a_offset_dim];
        
        if (i < b_offset_dim) b_broadcast_strides[i] = 0;
        else b_broadcast_strides[i] = (b_shape[i - b_offset_dim] == 1) ? 0 : b_strides[i - b_offset_dim];
    }

    int64_t inner_dim = ndim - 1;
    int64_t inner_size = out_shape[inner_dim];
    int64_t num_rows = numel / inner_size;

    int64_t s_a = a_broadcast_strides[inner_dim];
    int64_t s_b = b_broadcast_strides[inner_dim];
    int64_t s_out = out_strides[inner_dim];
    bool inner_fast = (s_a == 1 && s_b == 1 && s_out == 1);

    #pragma omp parallel for if(num_rows > 32)
    for (int64_t row = 0; row < num_rows; ++row) {
        int64_t temp = row;
        int64_t a_base = 0;
        int64_t b_base = 0;
        int64_t out_base = 0;
        
        for (int d = inner_dim - 1; d >= 0; --d) {
            int64_t size_d = out_shape[d];
            int64_t coord = temp % size_d;
            temp /= size_d;
            
            a_base += coord * a_broadcast_strides[d];
            b_base += coord * b_broadcast_strides[d];
            out_base += coord * out_strides[d];
        }

        if (inner_fast) {
            const T* ptr_a = a_data + a_base;
            const T* ptr_b = b_data + b_base;
            bool* ptr_out = out_data + out_base;
            for (int64_t i = 0; i < inner_size; ++i) {
                ptr_out[i] = check_isclose(get_val<T>(ptr_a, i), get_val<T>(ptr_b, i), rtol, atol, equal_nan);
            }
        } else {
            for (int64_t i = 0; i < inner_size; ++i) {
                double val_a = get_val<T>(a_data, a_base + i * s_a);
                double val_b = get_val<T>(b_data, b_base + i * s_b);
                out_data[out_base + i * s_out] = check_isclose(val_a, val_b, rtol, atol, equal_nan);
            }
        }
    }
}

void calculate(Tensor a, Tensor b, double rtol, double atol, bool equal_nan, Tensor out) {
    DataType dtype = a->dtype();
    if (dtype == DataType::F32) {
        isclose_kernel<float>(a, b, rtol, atol, equal_nan, out);
    } else if (dtype == DataType::F16) {
        isclose_kernel<fp16_t>(a, b, rtol, atol, equal_nan, out);
    } else if (dtype == DataType::BF16) {
        isclose_kernel<uint16_t>(a, b, rtol, atol, equal_nan, out);
    } else {
        throw std::runtime_error("Unsupported dtype for isclose");
    }
}

static bool registered = []() {
    IsClose::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::isclose_impl::cpu