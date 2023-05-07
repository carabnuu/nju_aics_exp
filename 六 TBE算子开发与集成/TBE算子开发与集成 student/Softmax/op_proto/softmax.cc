#include "softmax.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(SoftmaxInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Softmax, SoftmaxVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Softmax, SoftmaxInferShape);
VERIFY_FUNC_REG(Softmax, SoftmaxVerify);

}  // namespace ge
