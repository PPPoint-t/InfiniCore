#ifndef __CAST_H__
#define __CAST_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::cast::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        CastInfo _info;                                   \
        size_t _min_workspace_size;                       \
                                                          \
        Descriptor(                                       \
            CastInfo info,                                \
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
            infiniopTensorDescriptor_t in_desc);          \
                                                          \
        size_t minWorkspaceSize() const;                  \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *output,                                 \
            const void *input,                            \
            void *stream) const;                          \
    };                                                    \
    }

namespace op::cast {

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
        CastInfo info,
        void *workspace, size_t workspace_size,
        void *output, const void *input,
        void *stream) {
    
#define CASE_OUT(DT_OUT, TOUT)                                      \
    case DT_OUT:                                                    \
        switch (info.dt_in) {                                       \
        case INFINI_DTYPE_I32:                                      \
            algo.template run<TOUT, int32_t>(                       \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        case INFINI_DTYPE_I64:                                      \
            algo.template run<TOUT, int64_t>(                       \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        case INFINI_DTYPE_U32:                                      \
            algo.template run<TOUT, uint32_t>(                      \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        case INFINI_DTYPE_U64:                                      \
            algo.template run<TOUT, uint64_t>(                      \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        case INFINI_DTYPE_F16:                                      \
            algo.template run<TOUT, fp16_t>(                        \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        case INFINI_DTYPE_F32:                                      \
            algo.template run<TOUT, float>(                         \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        case INFINI_DTYPE_F64:                                      \
            algo.template run<TOUT, double>(                        \
                workspace, workspace_size, output, input, info.n, info, stream); \
            break;                                                  \
        default:                                                   \
            std::abort();                                           \
        }                                                          \
        break

        switch (info.dt_out) {
            CASE_OUT(INFINI_DTYPE_I32, int32_t);
            CASE_OUT(INFINI_DTYPE_I64, int64_t);
            CASE_OUT(INFINI_DTYPE_U32, uint32_t);
            CASE_OUT(INFINI_DTYPE_U64, uint64_t);
            CASE_OUT(INFINI_DTYPE_F16, fp16_t);
            CASE_OUT(INFINI_DTYPE_F32, float);
            CASE_OUT(INFINI_DTYPE_F64, double);
        default:
            std::abort();
        }

#undef CASE_OUT

        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::cast

#endif // __CAST_H__
