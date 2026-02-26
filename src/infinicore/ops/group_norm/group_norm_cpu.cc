#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/group_norm.hpp"
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>
#include <array>

namespace infinicore::op::group_norm_impl::cpu {

constexpr int64_t OMP_THRESHOLD = 4096;
constexpr int MAX_DIMS = 12;

inline float bf16_to_f32(uint16_t val) {
    union {
        uint32_t i;
        float f;
    } u;
    u.i = static_cast<uint32_t>(val) << 16;
    return u.f;
}

template <typename T>
inline double load_val(const void* ptr, int64_t offset, DataType dtype) {
    if (dtype == DataType::BF16) {
        return static_cast<double>(bf16_to_f32(reinterpret_cast<const uint16_t*>(ptr)[offset]));
    } else {
        return utils::cast<double>(reinterpret_cast<const T*>(ptr)[offset]);
    }
}

template <typename T>
inline void store_val(void* ptr, int64_t offset, double val, DataType dtype) {
    if (dtype == DataType::BF16) {
        float res_f = static_cast<float>(val);
        union {
            float f;
            uint32_t i;
        } u;
        u.f = res_f;
        uint16_t bf16_val = static_cast<uint16_t>(u.i >> 16);
        reinterpret_cast<uint16_t*>(ptr)[offset] = bf16_val;
    } else {
        reinterpret_cast<T*>(ptr)[offset] = utils::cast<T>(val);
    }
}

bool is_spatial_contiguous(const std::vector<size_t>& shape, const std::vector<int64_t>& strides, int start_dim) {
    int64_t expected_stride = 1;
    for (int i = shape.size() - 1; i >= start_dim; --i) {
        if (strides[i] != expected_stride) return false;
        expected_stride *= static_cast<int64_t>(shape[i]);
    }
    return true;
}

template <typename T>
void group_norm_kernel(const Tensor &input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps, Tensor &output) {
    auto shape = input->shape();
    auto strides = input->strides();
    auto out_strides = output->strides();
    
    int64_t N = shape[0];
    int64_t C = shape[1];
    
    int64_t HxW = 1;
    for (size_t i = 2; i < shape.size(); ++i) HxW *= shape[i];
    
    int64_t channels_per_group = C / num_groups;
    int64_t group_size = channels_per_group * HxW;
    int64_t total_elements = input->numel();

    const void* input_raw = input->data();
    void* output_raw = output->data();
    DataType dtype = input->dtype();
    
    const void* weight_raw = (weight.has_value() && weight.value()) ? weight.value()->data() : nullptr;
    const void* bias_raw = (bias.has_value() && bias.value()) ? bias.value()->data() : nullptr;

    int64_t w_stride = (weight_raw) ? weight.value()->strides()[0] : 0;
    int64_t b_stride = (bias_raw) ? bias.value()->strides()[0] : 0;

    bool input_spatial_contig = is_spatial_contiguous(shape, strides, 2);
    bool output_spatial_contig = is_spatial_contiguous(shape, out_strides, 2);
    bool use_fast_path = input_spatial_contig && output_spatial_contig;

    int64_t spatial_dims[MAX_DIMS];
    int64_t spatial_strides[MAX_DIMS];
    int64_t spatial_out_strides[MAX_DIMS];
    int spatial_rank = 0;
    
    if (!use_fast_path) {
        spatial_rank = shape.size() - 2;
        if (spatial_rank > MAX_DIMS) throw std::runtime_error("Dimension too large for stack allocation");
        for (size_t i = 2; i < shape.size(); ++i) {
            spatial_dims[i-2] = shape[i];
            spatial_strides[i-2] = strides[i];
            spatial_out_strides[i-2] = out_strides[i];
        }
    }

    #pragma omp parallel for collapse(2) if(total_elements > OMP_THRESHOLD)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            
            double sum = 0.0;
            double sum_sq = 0.0;

            for (int64_t c_g = 0; c_g < channels_per_group; ++c_g) {
                int64_t c = g * channels_per_group + c_g;
                int64_t base_offset = n * strides[0] + c * strides[1];

                if (use_fast_path) {
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    double sq0 = 0, sq1 = 0, sq2 = 0, sq3 = 0;
                    int64_t i = 0;

                    if (dtype == DataType::BF16) {
                        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(input_raw) + base_offset;
                        for (; i <= HxW - 4; i += 4) {
                            double v0 = static_cast<double>(bf16_to_f32(ptr[i]));
                            double v1 = static_cast<double>(bf16_to_f32(ptr[i+1]));
                            double v2 = static_cast<double>(bf16_to_f32(ptr[i+2]));
                            double v3 = static_cast<double>(bf16_to_f32(ptr[i+3]));
                            
                            s0 += v0; sq0 += v0*v0;
                            s1 += v1; sq1 += v1*v1;
                            s2 += v2; sq2 += v2*v2;
                            s3 += v3; sq3 += v3*v3;
                        }
                        for (; i < HxW; ++i) {
                            double v = static_cast<double>(bf16_to_f32(ptr[i]));
                            s0 += v; sq0 += v*v;
                        }
                    } else {
                        const T* ptr = reinterpret_cast<const T*>(input_raw) + base_offset;
                        for (; i <= HxW - 4; i += 4) {
                            double v0 = utils::cast<double>(ptr[i]);
                            double v1 = utils::cast<double>(ptr[i+1]);
                            double v2 = utils::cast<double>(ptr[i+2]);
                            double v3 = utils::cast<double>(ptr[i+3]);

                            s0 += v0; sq0 += v0*v0;
                            s1 += v1; sq1 += v1*v1;
                            s2 += v2; sq2 += v2*v2;
                            s3 += v3; sq3 += v3*v3;
                        }
                        for (; i < HxW; ++i) {
                            double v = utils::cast<double>(ptr[i]);
                            s0 += v; sq0 += v*v;
                        }
                    }
                    sum += (s0 + s1 + s2 + s3);
                    sum_sq += (sq0 + sq1 + sq2 + sq3);

                } else {
                    int64_t indices[MAX_DIMS] = {0};
                    int64_t current_offset = 0; 

                    for (int64_t i = 0; i < HxW; ++i) {
                        double val = load_val<T>(input_raw, base_offset + current_offset, dtype);
                        sum += val;
                        sum_sq += val * val;

                        for (int d = spatial_rank - 1; d >= 0; --d) {
                            indices[d]++;
                            current_offset += spatial_strides[d];
                            if (indices[d] < spatial_dims[d]) break;
                            indices[d] = 0;
                            current_offset -= spatial_dims[d] * spatial_strides[d];
                        }
                    }
                }
            }

            double mean = sum / group_size;
            double var = (sum_sq / group_size) - (mean * mean);
            if (var < 0) var = 0;
            double inv_std = 1.0 / std::sqrt(var + eps);

            for (int64_t c_g = 0; c_g < channels_per_group; ++c_g) {
                int64_t c = g * channels_per_group + c_g;
                
                double w = 1.0;
                double b = 0.0;
                if (weight_raw) w = load_val<T>(weight_raw, c * w_stride, dtype);
                if (bias_raw) b = load_val<T>(bias_raw, c * b_stride, dtype);
                
                double scale = inv_std * w;
                double shift = b - mean * scale;

                int64_t in_base = n * strides[0] + c * strides[1];
                int64_t out_base = n * out_strides[0] + c * out_strides[1];

                if (use_fast_path) {
                    int64_t i = 0;
                    if (dtype == DataType::BF16) {
                        const uint16_t* in_ptr = reinterpret_cast<const uint16_t*>(input_raw) + in_base;
                        uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output_raw) + out_base;
                        
                        for (; i <= HxW - 4; i += 4) {
                             auto process = [&](int idx) {
                                float val = bf16_to_f32(in_ptr[idx]);
                                float res = val * static_cast<float>(scale) + static_cast<float>(shift);
                                union { float f; uint32_t i; } u; u.f = res;
                                out_ptr[idx] = static_cast<uint16_t>(u.i >> 16);
                             };
                             process(i); process(i+1); process(i+2); process(i+3);
                        }
                        for (; i < HxW; ++i) {
                            float val = bf16_to_f32(in_ptr[i]);
                            float res = val * static_cast<float>(scale) + static_cast<float>(shift);
                            union { float f; uint32_t i; } u; u.f = res;
                            out_ptr[i] = static_cast<uint16_t>(u.i >> 16);
                        }

                    } else {
                        const T* in_ptr = reinterpret_cast<const T*>(input_raw) + in_base;
                        T* out_ptr = reinterpret_cast<T*>(output_raw) + out_base;
                        
                        for (; i <= HxW - 4; i += 4) {
                            double v0 = utils::cast<double>(in_ptr[i]);
                            double v1 = utils::cast<double>(in_ptr[i+1]);
                            double v2 = utils::cast<double>(in_ptr[i+2]);
                            double v3 = utils::cast<double>(in_ptr[i+3]);
                            
                            out_ptr[i]   = utils::cast<T>(v0 * scale + shift);
                            out_ptr[i+1] = utils::cast<T>(v1 * scale + shift);
                            out_ptr[i+2] = utils::cast<T>(v2 * scale + shift);
                            out_ptr[i+3] = utils::cast<T>(v3 * scale + shift);
                        }
                        for (; i < HxW; ++i) {
                            double val = utils::cast<double>(in_ptr[i]);
                            out_ptr[i] = utils::cast<T>(val * scale + shift);
                        }
                    }
                } else {
                    int64_t indices[MAX_DIMS] = {0};
                    int64_t in_offset_rel = 0;
                    int64_t out_offset_rel = 0;

                    for (int64_t i = 0; i < HxW; ++i) {
                        double val = load_val<T>(input_raw, in_base + in_offset_rel, dtype);
                        store_val<T>(output_raw, out_base + out_offset_rel, val * scale + shift, dtype);

                        for (int d = spatial_rank - 1; d >= 0; --d) {
                            indices[d]++;
                            in_offset_rel += spatial_strides[d];
                            out_offset_rel += spatial_out_strides[d];
                            if (indices[d] < spatial_dims[d]) break;
                            indices[d] = 0;
                            in_offset_rel -= spatial_dims[d] * spatial_strides[d];
                            out_offset_rel -= spatial_dims[d] * spatial_out_strides[d];
                        }
                    }
                }
            }
        }
    }
}

void calculate(Tensor input, int64_t num_groups, std::optional<Tensor> weight, std::optional<Tensor> bias, double eps, Tensor output) {
    auto dtype = input->dtype();
    if (dtype == DataType::F32) {
        group_norm_kernel<float>(input, num_groups, weight, bias, eps, output);
    } else if (dtype == DataType::F16) {
        group_norm_kernel<fp16_t>(input, num_groups, weight, bias, eps, output);
    } else if (dtype == DataType::BF16) {
        group_norm_kernel<uint16_t>(input, num_groups, weight, bias, eps, output);
    } else if (dtype == DataType::F64) {
        group_norm_kernel<double>(input, num_groups, weight, bias, eps, output);
    } else {
        throw std::runtime_error("Unsupported dtype for group_norm");
    }
}

static bool registered = []() {
    GroupNorm::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::group_norm_impl::cpu