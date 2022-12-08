import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("softmax")
def softmax_compute(x, y, axis, kernel_name="softmax"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    res = tbe.XXX(x)
    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def softmax(x, y, axis, kernel_name="softmax"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")

    res = softmax_compute(data_x, y, axis, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    tbe.build(schedule, config)
    