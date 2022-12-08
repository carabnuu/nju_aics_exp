import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("batch_normalization_dsl")
def batch_normalization_dsl_compute(x, scale, offset, mean, variance, y, batch_mean, batch_variance, epsilon, is_training, kernel_name="batch_normalization_dsl"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    res = tbe.XXX(x, scale, offset, mean, variance)
    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_normalization_dsl(x, scale, offset, mean, variance, y, batch_mean, batch_variance, epsilon, is_training, kernel_name="batch_normalization_dsl"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_scale = tvm.placeholder(scale.get("shape"), dtype=scale.get("dtype"), name="data_scale")
    data_offset = tvm.placeholder(offset.get("shape"), dtype=offset.get("dtype"), name="data_offset")
    data_mean = tvm.placeholder(mean.get("shape"), dtype=mean.get("dtype"), name="data_mean")
    data_variance = tvm.placeholder(variance.get("shape"), dtype=variance.get("dtype"), name="data_variance")

    res = batch_normalization_dsl_compute(data_x, data_scale, data_offset, data_mean, data_variance, y, batch_mean, batch_variance, epsilon, is_training, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_scale, data_offset, data_mean, data_variance, res]}
    tbe.build(schedule, config)
    