#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/rrelu.hpp"
#include <cmath>
#include <omp.h>

namespace infinicore::op::rrelu_impl::cpu {

inline float bf16_to_f32(uint16_t val) {
    union { uint32_t i; float f; } u;
    u.i = static_cast<uint32_t>(val) << 16;
    return u.f;
}

inline uint16_t f32_to_bf16(float val) {
    union { float f; uint32_t i; } u;
    u.f = val;
    return static_cast<uint16_t>(u.i >> 16);
}

template <typename T>
inline T compute_rrelu(T val, T slope) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return (val >= 0) ? val : static_cast<T>(val * slope);
    } else {
        float f_val = utils::cast<float>(val);
        float f_slope = utils::cast<float>(slope);
        float res = (f_val >= 0) ? f_val : (f_val * f_slope);
        return utils::cast<T>(res);
    }
}

template <>
inline uint16_t compute_rrelu<uint16_t>(uint16_t val, uint16_t slope) {
    float f_val = bf16_to_f32(val);
    float f_slope = bf16_to_f32(slope);
    float res = (f_val >= 0) ? f_val : (f_val * f_slope);
    return f32_to_bf16(res);
}

template <typename T>
void rrelu_kernel(const Tensor& input, double lower, double upper, bool training, Tensor& output) {
    int64_t numel = input->numel();
    if (numel == 0) return;

    double slope_val = (lower + upper) / 2.0;
    
    T slope_t;
    if constexpr (std::is_same_v<T, uint16_t>) {
        slope_t = f32_to_bf16(static_cast<float>(slope_val));
    } else {
        slope_t = utils::cast<T>(slope_val);
    }

    const T* in_ptr = reinterpret_cast<const T*>(input->data());
    T* out_ptr = reinterpret_cast<T*>(output->data());

    if (input->is_contiguous() && output->is_contiguous()) {
        #pragma omp parallel for if(numel > 4096)
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_rrelu(in_ptr[i], slope_t);
        }
        return;
    }

    int ndim = input->ndim();
    auto shape = input->shape();
    auto in_strides = input->strides();
    auto out_strides = output->strides();

    int64_t inner_dim = ndim - 1;
    int64_t inner_size = shape[inner_dim];
    int64_t num_rows = numel / inner_size;

    int64_t s_in = in_strides[inner_dim];
    int64_t s_out = out_strides[inner_dim];
    bool inner_fast = (s_in == 1 && s_out == 1);

    #pragma omp parallel for if(num_rows > 32)
    for (int64_t row = 0; row < num_rows; ++row) {
        int64_t temp = row;
        int64_t in_base = 0;
        int64_t out_base = 0;
        
        for (int d = inner_dim - 1; d >= 0; --d) {
            int64_t size_d = shape[d];
            int64_t coord = temp % size_d;
            temp /= size_d;
            in_base += coord * in_strides[d];
            out_base += coord * out_strides[d];
        }

        if (inner_fast) {
            const T* p_in = in_ptr + in_base;
            T* p_out = out_ptr + out_base;
            for (int64_t i = 0; i < inner_size; ++i) {
                p_out[i] = compute_rrelu(p_in[i], slope_t);
            }
        } else {
            for (int64_t i = 0; i < inner_size; ++i) {
                out_ptr[out_base + i * s_out] = compute_rrelu(in_ptr[in_base + i * s_in], slope_t);
            }
        }
    }
}

void calculate(Tensor input, double lower, double upper, bool training, Tensor output) {
    if (training) {
        throw std::runtime_error("RReLU training mode not supported on CPU yet.");
    }

    DataType dtype = input->dtype();
    if (dtype == DataType::F32) {
        rrelu_kernel<float>(input, lower, upper, training, output);
    } else if (dtype == DataType::F16) {
        rrelu_kernel<fp16_t>(input, lower, upper, training, output);
    } else if (dtype == DataType::BF16) {
        rrelu_kernel<uint16_t>(input, lower, upper, training, output);
    } else {
        throw std::runtime_error("Unsupported dtype for rrelu");
    }
}

static bool registered = []() {
    RReLU::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::rrelu_impl::cpu