#include "../cuda/kernel.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../cast.h"
#include "cast_nvidia.cuh"
#include "../info.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstring>

namespace op::cast::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

template <typename T> struct MapCudaType { using Type = T; };
template <> struct MapCudaType<fp16_t> { using Type = half; };

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info_r = CastInfo::create(out_desc, in_desc);
    CHECK_RESULT(info_r);
    auto info = info_r.take();

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        info,
        workspace_size,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

struct Algo {
    int block_size;

    Algo(int bs = 256) : block_size(bs) {}
    template <class ToutHost, class TinHost>
    infiniStatus_t run(
        void *workspace, size_t workspace_size,
        void *output_, const void *input_, size_t n,
        const op::cast::CastInfo &info, void *stream_) const {
        int bs = 0, grid = 0;
        using DevTout = typename MapCudaType<ToutHost>::Type;
        using DevTin  = typename MapCudaType<TinHost>::Type;

        auto out_dev = reinterpret_cast<DevTout *>(output_);
        auto in_dev  = reinterpret_cast<const DevTin *>(input_);
        auto stream = reinterpret_cast<cudaStream_t>(stream_);

        int ndim = static_cast<int>(info.shape.size());
        if (ndim == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        std::vector<size_t> h_shape(info.shape.begin(), info.shape.end());
        std::vector<size_t> h_div(ndim);
        h_div[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            h_div[d] = h_div[d + 1] * h_shape[d + 1];
        }

        std::vector<long long> h_in_stride(ndim), h_out_stride(ndim);
        for (int d = 0; d < ndim; ++d) {
            h_in_stride[d] = static_cast<long long>(info.in_stride[d]);
            h_out_stride[d] = static_cast<long long>(info.out_stride[d]);
        }

        size_t *d_shape = nullptr;
        size_t *d_div = nullptr;
        long long *d_in_stride = nullptr;
        long long *d_out_stride = nullptr;

        cudaError_t err = cudaSuccess;
        err = cudaMalloc(reinterpret_cast<void **>(&d_shape), sizeof(size_t) * ndim);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(reinterpret_cast<void **>(&d_div), sizeof(size_t) * ndim);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(reinterpret_cast<void **>(&d_in_stride), sizeof(long long) * ndim);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(reinterpret_cast<void **>(&d_out_stride), sizeof(long long) * ndim);
        if (err != cudaSuccess) goto cleanup;

        err = cudaMemcpyAsync(d_shape, h_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(d_div, h_div.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(d_in_stride, h_in_stride.data(), sizeof(long long) * ndim, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpyAsync(d_out_stride, h_out_stride.data(), sizeof(long long) * ndim, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;

        bs = block_size > 0 ? block_size : 256;
        grid = static_cast<int>((n + bs - 1) / bs);
        if (grid <= 0) grid = 1;

        cast_kernel<DevTout, DevTin><<<grid, bs, 0, stream>>>(
            out_dev, in_dev, n, d_shape, d_div, d_in_stride, d_out_stride, ndim);

        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) goto cleanup;

        cudaFree(d_shape);
        cudaFree(d_div);
        cudaFree(d_in_stride);
        cudaFree(d_out_stride);
        return INFINI_STATUS_SUCCESS;

    cleanup:
        cudaFree(d_shape);
        cudaFree(d_div);
        cudaFree(d_in_stride);
        cudaFree(d_out_stride);
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

};

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    // 禁止 inplace
    if (output == const_cast<void*>(input)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 使用 handle 的 block size 指定 kernel 参数
    auto block_size = _opaque->internal->blockSizeX();
    Calculate::calculate<Algo>(
        Algo{static_cast<int>(block_size)},
        _info, workspace, workspace_size,
        output, input, stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cast::nvidia
