#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/bitwise_xor.hpp"
#include <omp.h>
#include <vector>

namespace infinicore::op::bitwise_xor_impl::cpu {

template <typename T>
void bitwise_xor_kernel(const Tensor& a, const Tensor& b, Tensor& out) {
    int64_t numel = out->numel();
    if (numel == 0) return;

    const T* a_data = reinterpret_cast<const T*>(a->data());
    const T* b_data = reinterpret_cast<const T*>(b->data());
    T* out_data = reinterpret_cast<T*>(out->data());

    if (a->is_contiguous() && b->is_contiguous() && out->is_contiguous()) {
        #pragma omp parallel for if(numel > 4096)
        for (int64_t i = 0; i < numel; ++i) {
            out_data[i] = a_data[i] ^ b_data[i];
        }
        return;
    }

    int ndim = out->ndim();
    if (ndim == 0) {
        *out_data = *a_data ^ *b_data;
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

    int64_t a_inner_stride = a_broadcast_strides[inner_dim];
    int64_t b_inner_stride = b_broadcast_strides[inner_dim];
    int64_t out_inner_stride = out_strides[inner_dim];

    bool inner_contiguous = (a_inner_stride == 1 && b_inner_stride == 1 && out_inner_stride == 1);

    #pragma omp parallel for if(num_rows > 32)
    for (int64_t row = 0; row < num_rows; ++row) {
        int64_t temp_row = row;
        int64_t a_base = 0;
        int64_t b_base = 0;
        int64_t out_base = 0;

        for (int d = inner_dim - 1; d >= 0; --d) {
            int64_t size_d = out_shape[d];
            int64_t coord = temp_row % size_d;
            temp_row /= size_d;
            
            a_base += coord * a_broadcast_strides[d];
            b_base += coord * b_broadcast_strides[d];
            out_base += coord * out_strides[d];
        }

        if (inner_contiguous) {
            const T* ptr_a = a_data + a_base;
            const T* ptr_b = b_data + b_base;
            T* ptr_out = out_data + out_base;
            for (int64_t i = 0; i < inner_size; ++i) {
                ptr_out[i] = ptr_a[i] ^ ptr_b[i];
            }
        } else {
            for (int64_t i = 0; i < inner_size; ++i) {
                out_data[out_base + i * out_inner_stride] = 
                    a_data[a_base + i * a_inner_stride] ^ 
                    b_data[b_base + i * b_inner_stride];
            }
        }
    }
}

void calculate(Tensor a, Tensor b, Tensor out) {
    DataType dtype = a->dtype();
    if (dtype == DataType::I8) {
        bitwise_xor_kernel<int8_t>(a, b, out);
    } else if (dtype == DataType::I16) {
        bitwise_xor_kernel<int16_t>(a, b, out);
    } else if (dtype == DataType::I32) {
        bitwise_xor_kernel<int32_t>(a, b, out);
    } else if (dtype == DataType::I64) {
        bitwise_xor_kernel<int64_t>(a, b, out);
    } else if (dtype == DataType::U8) {
        bitwise_xor_kernel<uint8_t>(a, b, out);
    } else if (dtype == DataType::BOOL) {
        bitwise_xor_kernel<bool>(a, b, out);
    } else {
        throw std::runtime_error("Unsupported dtype for bitwise_xor");
    }
}

static bool registered = []() {
    BitwiseXor::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::bitwise_xor_impl::cpu