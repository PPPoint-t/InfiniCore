#include "../../../devices/nvidia/nvidia_common.cuh"
#include "leakyrelu_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

namespace op::leakyrelu::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    float negative_slope) {

    auto result = LeakyreluInfo::create(output_desc, input_desc, negative_slope);
    CHECK_RESULT(result);
    auto info = result.take();

    if (info.input_strides.back() != 1 || info.output_strides.back() != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        new Opaque{ reinterpret_cast<device::nvidia::Handle *>(handle)->internal() },
        std::move(info),
        0,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// launch kernel with different block sizes; pick runtimeBlock = min(BLOCK_SIZE, device_maxThreads, HARD_CAP)
// launch kernel with conservative compile-time BLOCK_SIZE and robust runtime caps
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    size_t numel,
    void *y, ptrdiff_t stride_y,
    const void *x, ptrdiff_t stride_x,
    size_t dim,
    infiniDtype_t atype,
    float negative_slope,
    cudaStream_t cuda_stream,
    uint32_t device_maxThreads,
    uint32_t device_maxBlocks) {

    // Much more conservative HARD_CAP to avoid any device quirks.
    // 256 是一个普遍安全且高效的选择。
    constexpr uint32_t HARD_CAP = 256u;

    uint32_t effective_device_maxThreads = device_maxThreads;
    if (effective_device_maxThreads == 0 || effective_device_maxThreads > HARD_CAP) {
        effective_device_maxThreads = HARD_CAP;
    }

    // choose runtime block size (<= compile-time BLOCK_SIZE and <= device cap)
    uint32_t runtimeBlock = static_cast<uint32_t>(std::min<size_t>(BLOCK_SIZE, effective_device_maxThreads));
    if (runtimeBlock == 0) return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    // number of blocks needed, cap by device_maxBlocks
    size_t blocks_needed = (numel + (size_t)runtimeBlock - 1) / static_cast<size_t>(runtimeBlock);
    uint32_t grid = static_cast<uint32_t>(std::min(blocks_needed, static_cast<size_t>(device_maxBlocks)));
    if (grid == 0) grid = 1;

    // Print diagnostics (stderr) to help debug resource issues
    std::fprintf(stderr, "[leakyrelu launch] BLOCK_SIZE(template)=%u effective_device_maxThreads=%u runtimeBlock=%u blocks_needed=%zu grid=%u device_maxBlocks=%u numel=%zu dim=%zu stride_x=%td stride_y=%td\n",
                 (unsigned)BLOCK_SIZE, effective_device_maxThreads, runtimeBlock, blocks_needed, grid, device_maxBlocks, numel, dim, stride_x, stride_y);

#define LAUNCH(Tdata, Tcompute) \
    leakyreluKernel<BLOCK_SIZE, Tcompute, Tdata><<<grid, runtimeBlock, 0, cuda_stream>>>( \
        reinterpret_cast<Tdata *>(y), stride_y, reinterpret_cast<const Tdata *>(x), stride_x, dim, numel, negative_slope)

    if (atype == INFINI_DTYPE_F16) {
        LAUNCH(half, float);
    } else if (atype == INFINI_DTYPE_BF16) {
        LAUNCH(__nv_bfloat16, float);
    } else if (atype == INFINI_DTYPE_F32) {
        LAUNCH(float, float);
    } else if (atype == INFINI_DTYPE_F64) {
        LAUNCH(double, double);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH

    // immediate check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "leakyrelu kernel launch failed: %s\n", cudaGetErrorString(err));
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    size_t numel = _info.numel();
    if (numel == 0) return INFINI_STATUS_SUCCESS;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    uint32_t device_maxBlocks = static_cast<uint32_t>(_opaque->internal->gridSizeX());
    uint32_t device_maxThreads = static_cast<uint32_t>(_opaque->internal->maxThreadsPerBlock());

    // Debug print device-reported capabilities
    std::fprintf(stderr, "[leakyrelu calc] device_maxThreads(raw)=%u device_maxBlocks=%u\n", device_maxThreads, device_maxBlocks);

    // Conservative hard cap used inside launchKernel (HARD_CAP=256)
    auto stride_x = _info.input_strides[0];
    auto stride_y = _info.output_strides[0];
    auto dim = _info.ndim() > 0 ? _info.shape.back() : 1;

    // Always instantiate a conservative compile-time BLOCK_SIZE (256)
    // This avoids heavy register/shmem usage that some template instantiations may cause.
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(numel, y, stride_y, x, stride_x, dim, _info.atype, _info.negative_slope, cuda_stream, device_maxThreads, device_maxBlocks));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(numel, y, stride_y, x, stride_x, dim, _info.atype, _info.negative_slope, cuda_stream, device_maxThreads, device_maxBlocks));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(numel, y, stride_y, x, stride_x, dim, _info.atype, _info.negative_slope, cuda_stream, device_maxThreads, device_maxBlocks));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}


} // namespace op::leakyrelu::nvidia
