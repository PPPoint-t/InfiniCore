#ifndef __LEAKYRELU_H__
#define __LEAKYRELU_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::leakyrelu::NAMESPACE {                  \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        LeakyReLUInfo _info;                              \
        size_t _min_workspace_size;                       \
                                                          \
        Descriptor(                                       \
            LeakyReLUInfo info,                           \
            size_t min_workspace_size,                    \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _info(info),                                \
              _min_workspace_size(min_workspace_size) {}  \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t out_desc,          \
            infiniopTensorDescriptor_t in_desc,           \
            float negative_slope);                        \
                                                          \
        size_t workspaceSize() const;                     \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *output,                                 \
            const void *input,                            \
            void *stream) const;                          \
    };                                                    \
    }

namespace op::leakyrelu {

struct CalculateArgs {
    void *workspace;
    size_t workspace_size;
    void *output;
    const void *input;
    void *stream;
};

class Calculate {
public:
    template <class Algo>
    static infiniStatus_t calculate(
        Algo algo,
        LeakyReLUInfo info,
        void *workspace, size_t workspace_size,
        void *output, const void *input,
        void *stream) {

#define CASE_DT(DT)                                           \
    case DT:                                                  \
        switch (info.dt_in) {                                    \
        case INFINI_DTYPE_F16:                                \
            algo.template run<fp16_t>(                        \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                            \
        case INFINI_DTYPE_BF16:                               \
            algo.template run<bf16_t>(                        \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                            \
        case INFINI_DTYPE_F32:                                \
            algo.template run<float>(                         \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                            \
        case INFINI_DTYPE_F64:                                \
            algo.template run<double>(                        \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                            \
        default:                                              \
            std::abort();                                     \
        }                                                     \
        break

        switch (info.dt_in) {
            CASE_DT(INFINI_DTYPE_F16);
            CASE_DT(INFINI_DTYPE_BF16);
            CASE_DT(INFINI_DTYPE_F32);
            CASE_DT(INFINI_DTYPE_F64);
        default:
            std::abort();
        }

#undef CASE_DT

        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::leakyrelu

#endif // __LEAKYRELU_H__
