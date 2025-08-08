#ifndef __LEAKYRELU_CUDA_KERNEL_CUH__
#define __LEAKYRELU_CUDA_KERNEL_CUH__

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
INFINIOP_CUDA_KERNEL leakyreluKernel(
    Tdata *__restrict__ y,
    ptrdiff_t stride_y,
    const Tdata *__restrict__ x,
    ptrdiff_t stride_x,
    size_t dim,
    size_t numel,
    float negative_slope) {

    // use runtime blockDim.x (bs) and gridDim.x for correct indexing
    const unsigned int bs = static_cast<unsigned int>(blockDim.x);
    const size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(bs) + static_cast<size_t>(threadIdx.x);
    const size_t stride = static_cast<size_t>(bs) * static_cast<size_t>(gridDim.x);

    for (size_t linear = tid; linear < numel; linear += stride) {
        size_t row = linear / dim;
        size_t col = linear % dim;
        size_t in_idx = row * static_cast<size_t>(stride_x) + col;
        size_t out_idx = row * static_cast<size_t>(stride_y) + col;

        Tcompute xin = (Tcompute)(x[in_idx]);
        Tcompute out = xin >= (Tcompute)0 ? xin : xin * (Tcompute)negative_slope;
        y[out_idx] = (Tdata)(out);
    }
}

#endif // __LEAKYRELU_CUDA_KERNEL_CUH__
