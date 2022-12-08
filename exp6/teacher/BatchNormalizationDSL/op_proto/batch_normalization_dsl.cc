#include "batch_normalization_dsl.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(BatchNormalizationDSLInferShape)
{
    // 获取输出数据描述
    TensorDesc tensordesc_output = op.GetOutputDescByName("x");
    tensordesc_output.SetShape(op.GetInputDescByName("x").GetShape());
    tensordesc_output.SetDataType(op.GetInputDescByName("x").GetDataType());
    tensordesc_output.SetFormat(op.GetInputDescByName("x").GetFormat());
    //直接将输入x的Tensor描述信息赋给输出
    (void)op.UpdateOutputDesc("y", tensordesc_output);


    // 将scale的Tensor描述信息赋给batch_mean和batch_variance
    TensorDesc tensordesc_output_mean = op.GetOutputDescByName("scale");
    tensordesc_output_mean.SetShape(op.GetInputDescByName("scale").GetShape());
    tensordesc_output_mean.SetDataType(op.GetInputDescByName("scale").GetDataType());
    tensordesc_output_mean.SetFormat(op.GetInputDescByName("scale").GetFormat());
    (void)op.UpdateOutputDesc("batch_mean", tensordesc_output_mean);
    (void)op.UpdateOutputDesc("batch_variance", tensordesc_output_mean);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(BatchNormalizationDSL, BatchNormalizationDSLVerify)
{
    if (op.GetInputDescByName("scale").GetDataType() != op.GetInputDescByName("offset").GetDataType()) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchNormalizationDSL, BatchNormalizationDSLInferShape);
VERIFY_FUNC_REG(BatchNormalizationDSL, BatchNormalizationDSLVerify);

}  // namespace ge
