from tbe import tvm
from tbe.common.register import register_op_compute
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util


def _output_data_y_compute(x, mean, variance, scale, offset, epsilon):
    # x = (x - mean)/(var + epsilon)**0.5
    # y = scale*x + offset
    shape_x = shape_util.shape_to_list(x.shape)
    y_add = tbe.vadds(variance, epsilon)
    y_sqrt = tbe.vsqrt(y_add)
    var_sub = tbe.vsub(x, mean)
    y_norm = tbe.vdiv(var_sub, y_sqrt)
    scale_broad = tbe.broadcast(scale, shape_x)
    offset_broad = tbe.broadcast(offset, shape_x)
    res = tbe.vadd(tbe.vmul(scale_broad, y_norm), offset_broad)

    return res


def _batch_norm_dsl_train_compute(x, scale, offset, epsilon):
    # 如果输入x的dtype为float16，则转为float32进行计算, 防止训练的时候下溢造成训练困难
    is_cast = False
    if x.dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        is_cast = True
        x = tbe.cast_to(x, "float32")

    shape_x = shape_util.shape_to_list(x.shape)

    axis = [0, 2, 3]    # NHW
    num = shape_x[0]*shape_x[2]*shape_x[3]
    num_rec = 1.0/num

    # 根据 x 的维度 C 计算均值
    mean_sum = tbe.sum(x, axis, True)
    mean_muls = tbe.vmuls(mean_sum, num_rec)
    mean_broadcast = tbe.broadcast(mean_muls, shape_x)

    # 根据 x 的维度 C 计算方差
    var_sub = tbe.vsub(x, mean_broadcast)
    var_mul = tbe.vmul(var_sub, var_sub)
    var_sum = tbe.sum(var_mul, axis, True)
    var_muls = tbe.vmuls(var_sum, num_rec)
    var_broadcast = tbe.broadcast(var_muls, shape_x)

    # BatchNormalization 计算
    res_y = _output_data_y_compute(x, mean_broadcast, var_broadcast, scale, offset, epsilon)

    # 保持输入输出的dtype一致
    if is_cast:
        res_y = tbe.cast_to(res_y, "float16")

    # 计算当前batch的均值和方差 NCHW -> C
    res_batch_mean = tbe.vmuls(mean_sum, num_rec)
    if num == 1:
        batch_var_scaler = 0.0
    else:
        batch_var_scaler = float(num)/(num - 1)
    res_batch_var = tbe.vmuls(var_muls, batch_var_scaler)
    res = [res_y, res_batch_mean, res_batch_var]
    return res


def _batch_norm_dsl_inf_compute(x, scale, offset, mean, variance, epsilon):
    # 如果输入x的dtype为float16，则转为float32进行计算
    is_cast = False
    if x.dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        is_cast = True
        x = tbe.cast_to(x, "float32")
    shape_x = shape_util.shape_to_list(x.shape)

    # 将已有的均值和方差广播至同纬度
    mean_broadcast = tbe.broadcast(mean, shape_x)
    var_broadcast = tbe.broadcast(variance, shape_x)

    # BatchNormalization 计算
    res_y = _output_data_y_compute(x, mean_broadcast, var_broadcast, scale, offset, epsilon)

    # 保持输入输出的dtype一致
    if is_cast:
        res_y = tbe.cast_to(res_y, "float16")

    # 推理的batch_mean和batch_var 即为传入的mean和variance
    scaler_zero = 0.0
    res_batch_mean = tbe.vadds(mean, scaler_zero)
    res_batch_var = tbe.vadds(variance, scaler_zero)
    res = [res_y, res_batch_mean, res_batch_var]
    return res


@register_op_compute("batch_normalization_dsl")
def batch_normalization_dsl_compute(x, scale, offset, mean, variance, y, batch_mean, batch_variance, epsilon, is_training, kernel_name="batch_normalization_dsl"):
    if is_training:     # 训练时，均值和方差根据样本计算得到
        res = _batch_norm_dsl_train_compute(x, scale, offset, epsilon)
    else:               # 推理时，均值和方差需要传入
        res = _batch_norm_dsl_inf_compute(x, scale, offset, mean, variance, epsilon)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_normalization_dsl(x, scale, offset, mean, variance, y, batch_mean, batch_variance, epsilon, is_training, kernel_name="batch_normalization_dsl"):
    """
    x: 输入张量(N,C1,H,W,C0)
    scale: 缩放量(C1*C0,)
    offset: 偏移量(C1*C0,)
    mean: 均值(C1*C0,)
    variance: 方差(C1*C0,)
    y: 输出张量(N,C1,H,W,C0)
    batch_mean: 当前张量的均值(C1*C0,)
    batch_variance: 当前张量的方差(C1*C0,)
    epsilon: 防止方差为0导致除数为0
    is_training: 训练验证开关
    kernel_name: 算子名称
    """
    # 检查输入的dtype和shape
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")
    para_check.check_dtype(dtype_scale.lower(), ("float32", "float16"), param_name="scale")
    para_check.check_dtype(dtype_offset.lower(), ("float32", "float16"), param_name="offset")
    if not is_training:
        dtype_mean = mean.get("dtype")
        dtype_variance = variance.get("dtype")
        para_check.check_dtype(dtype_mean.lower(), ("float32", "float16"), param_name="mean")
        para_check.check_dtype(dtype_variance.lower(), ("float32", "float16"), param_name="variance")

    # (C1*C0,) -> (1,C1,1,1,C0)
    shape_scale = [1, shape_x[1], 1, 1, shape_x[4]]
    shape_offset = shape_scale
    if not is_training:
        shape_mean = shape_scale
        shape_variance = shape_scale

    # 构建输入节点, 使用TVM的placeholder接口对输入tensor进行占位，返回tensor对象
    data_x = tvm.placeholder(shape_x, dtype=dtype_x.lower(), name="data_x")
    data_scale = tvm.placeholder(shape_scale, dtype=dtype_scale.lower(), name="data_scale")
    data_offset = tvm.placeholder(shape_offset, dtype=dtype_offset.lower(), name="data_offset")
    if is_training:
        data_mean, data_variance = None, None
    else:
        data_mean = tvm.placeholder(shape_mean, dtype=dtype_mean, name="data_mean")
        data_variance = tvm.placeholder(shape_variance, dtype=dtype_variance, name="data_variance")

    # BatchNormalization计算
    res = batch_normalization_dsl_compute(data_x, data_scale, data_offset, 
                                          data_mean, data_variance, y, 
                                          batch_mean, batch_variance, epsilon, 
                                          is_training, kernel_name)

    # 自动调度
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    if is_training:
        tensor_list = [data_x, data_scale, data_offset] + list(res)
    else:
        tensor_list = [data_x, data_scale, data_offset,
                       data_mean, data_variance] + list(res)

    # 编译配置
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedule, config)
