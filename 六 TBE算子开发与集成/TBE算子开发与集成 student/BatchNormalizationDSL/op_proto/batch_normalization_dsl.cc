#include "batch_normalization_dsl.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(BatchNormalizationDSLInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(BatchNormalizationDSL, BatchNormalizationDSLVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchNormalizationDSL, BatchNormalizationDSLInferShape);
VERIFY_FUNC_REG(BatchNormalizationDSL, BatchNormalizationDSLVerify);

}  // namespace ge
