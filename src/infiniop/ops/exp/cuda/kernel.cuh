#ifndef __EXP_CUDA_H__
#define __EXP_CUDA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace op::exp::cuda {
typedef struct ExpOp {
  static constexpr size_t num_inputs = 1;

  template <typename T>
  __device__ __forceinline__ T operator()(const T &x) const {
    if constexpr (std::is_same_v<T, half2>) {
        float2 vf = __half22float2(x);
        float2 vr = make_float2(__expf(vf.x), __expf(vf.y));
        return __float22half2_rn(vr);
    } else if constexpr (std::is_same_v<T, half>) {
        float xf = __half2float(x);
        return __float2half_rn(__expf(xf));
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
        float f0 = __bfloat162float(__low2bfloat16(x));
        float f1 = __bfloat162float(__high2bfloat16(x));
        return __floats2bfloat162_rn(__expf(f0), __expf(f1));
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        float xf = __bfloat162float(x);
        return __float2bfloat16_rn(__expf(xf));
    } else if constexpr (std::is_same_v<T, float>) {
        return __expf(x);
    } else if constexpr (std::is_same_v<T, double>) {
        return std::exp(x);
    } else {
        return std::exp(x);
    }
  }
} ExpOp;
} // namespace

#endif // __EXP_CUDA_H__
