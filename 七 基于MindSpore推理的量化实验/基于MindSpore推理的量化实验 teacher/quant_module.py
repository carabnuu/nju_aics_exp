import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


def calculate_scale_zero_point(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = float((max_val - min_val) / (qmax - qmin))  # S=(rmax-rmin)/(qmax-qmin)
    zero_point = round(qmax - max_val / scale)  # Z=round(qmax-rmax/scale)

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax

    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    # TODO 请根据公式实现张量的量化操作
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    q_x = zero_point + x / scale
    q_x = (q_x.clip(qmin, qmax)).round()  # q=round(r/S+Z)
    return q_x.astype(ms.float32)       # 由于mindspore不支持int类型的运算，因此我们还是用float来表示整数


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)  # r=S(q-Z)


class QParam(object):
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None

    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        self.max = 0 if self.max < 0 else self.max

        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        self.min = 0 if self.min > 0 else self.min

        self.scale, self.zero_point = calculate_scale_zero_point(self.min, self.max, self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)


class QModule(nn.Cell):
    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class QConv2d(QModule):
    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits)
        self.M = None

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # TODO 请实现卷积模块权重参数量化
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.conv_module.weight = self.qw.quantize_tensor(self.conv_module.weight)
        self.conv_module.weight = self.conv_module.weight - self.qw.zero_point

        self.conv_module.bias = quantize_tensor(self.conv_module.bias,
                                                scale=self.qi.scale * self.qw.scale,
                                                zero_point=0, num_bits=32, signed=True)

    def construct(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quantize_tensor(x)
            x = self.qi.dequantize_tensor(x)

        self.qw.update(self.conv_module.weight)
        self.conv_module.weight = self.qw.quantize_tensor(self.conv_module.weight)
        self.conv_module.weight = self.qw.dequantize_tensor(self.conv_module.weight)
        x = ops.conv2d(x, self.conv_module.weight, stride=self.conv_module.stride, pad_mode=self.conv_module.pad_mode)
        if self.conv_module.bias is not None:
            x = ops.bias_add(x, self.conv_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.quantize_tensor(x)
            x = self.qo.dequantize_tensor(x)
        return x

    def quantize_inference(self, x):
        # TODO 请实现卷积模块量化推理
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clip(0., 2. ** self.num_bits - 1.).round()
        return x


class QDense(QModule):
    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QDense, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)
        self.M = ms.Tensor([])

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # TODO 请实现全连接模块权重参数量化
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.fc_module.weight = self.qw.quantize_tensor(self.fc_module.weight)
        self.fc_module.weight = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias = quantize_tensor(self.fc_module.bias,
                                              scale=self.qi.scale * self.qw.scale,
                                              zero_point=0,
                                              num_bits=32,
                                              signed=True)

    def construct(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quantize_tensor(x)
            x = self.qi.dequantize_tensor(x)

        self.qw.update(self.fc_module.weight)
        self.fc_module.weight = self.qw.quantize_tensor(self.fc_module.weight)
        self.fc_module.weight = self.qw.dequantize_tensor(self.fc_module.weight)
        x = ops.matmul(x, self.fc_module.weight.T)
        x = ops.bias_add(x, self.fc_module.bias)
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.quantize_tensor(x)
            x = self.qo.dequantize_tensor(x)
        return x

    def quantize_inference(self, x):
        # TODO 请实现全连接模块量化推理
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x = x.round()
        x = x + self.qo.zero_point
        x = x.clip(0., 2. ** self.num_bits - 1.).round()
        return x


class QReLU(QModule):
    def __init__(self, qi=False, num_bits=None):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def construct(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quantize_tensor(x)
            x = self.qi.dequantize_tensor(x)
        x = ops.relu(x)

        return x

    def quantize_inference(self, x):
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x


class QMaxPooling2d(QModule):
    def __init__(self, max_pool_module, qi=False, num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.max_pool_module = max_pool_module

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def construct(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.quantize_tensor(x)
            x = self.qi.dequantize_tensor(x)
        x = self.max_pool_module(x)
        return x

    def quantize_inference(self, x):
        return self.max_pool_module(x)
