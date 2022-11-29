import numpy as np
from numba import jit

class ConvolutionLayer(object):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        # 输入通道数
        self.in_channels = in_channels
        # 输出通道数
        self.out_channels = out_channels
        # 卷积核尺寸
        self.kernel_size = kernel_size
        # 步长
        self.stride = stride
        # 填充长度
        self.padding = padding

        # 卷积核权重
        self.weight = np.random.normal(loc=0.0, scale=0.01,
                                       size=(self.out_channels, self.in_channels,
                                             self.kernel_size, self.kernel_size))
        # 卷积核偏置
        self.bias = np.zeros([self.out_channels])

        # 输入张量的形状，用于反向传播
        self.input_shape = None

    def forward(self, inputs):
        # 记录输入张量的形状，inputs: (N,C,H,W)
        self.input_shape = inputs.shape
        batch, channel, height, width = inputs.shape

        # 获取输入张量填充后的宽高
        pad_height = height + self.padding * 2
        pad_width = width + self.padding * 2

        # 将输入张量进行填充
        inputs_pad = np.zeros((batch, channel, pad_height, pad_width), dtype=inputs.dtype)
        inputs_pad[:, :, self.padding:height + self.padding, self.padding:width + self.padding] = inputs

        # 获取输出张量的宽高，并构建输出张量
        out_height = int((pad_height - self.kernel_size) / self.stride + 1)
        out_width = int((pad_width - self.kernel_size) / self.stride + 1)
        outputs = np.zeros((batch, self.out_channels, out_height, out_width), dtype=inputs.dtype)
        
        # 正向传播
        outputs = self._conv(inputs_pad, outputs, self.weight, self.bias, self.kernel_size, self.stride)
        return outputs

    def backward(self, out_grad):
        # 获得输入张量，填充后输入张量，输出张量的形状
        batch, channel, height, width = self.input_shape
        _, out_channel, out_height, out_width = out_grad.shape
        pad_height = height + self.padding * 2
        pad_width = width + self.padding * 2

        # 构建填充输入张量的梯度
        in_grad = np.zeros((batch, channel, pad_height, pad_width))

        # 反向传播
        in_grad = self._conv_back(out_grad, in_grad, self.weight, self.kernel_size, self.stride)
        
        # 返回输入张量梯度
        in_grad = in_grad[:, :, self.padding:height + self.padding, self.padding:width + self.padding]
        return in_grad

    def load_params(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    @staticmethod
    @jit(nopython=True)     # 可以将python函数编译为机器代码的JIT编译器，可以极大的加速for循环的运行速度
    def _conv(inputs_pad, outputs, weight, bias, kernel_size, stride):
        # TODO：根据公式编写下列代码 请用for循环实现
        in_channels = inputs_pad.shape[1]
        batch, out_channels, out_height, out_width = outputs.shape
        for n in range(batch):
            for c in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        hs, ws = h * stride, w * stride
                        val = 0
                        for k in range(in_channels):
                            for i in range(kernel_size):
                                for j in range(kernel_size):
                                    val += weight[c, k, i, j] * inputs_pad[n, k, hs+i, ws+j]
                        val += bias[c]
                        outputs[n, c, h, w] = val
        return outputs

    @staticmethod
    @jit(nopython=True)
    def _conv_back(out_grad, in_grad, weight, kernel_size, stride):
        # TODO：根据公式编写下列代码 请用for循环实现
        in_channels = in_grad.shape[1]
        batch, out_channel, out_height, out_width = out_grad.shape
        for n in range(batch):
            for h in range(out_height):
                for w in range(out_width):
                    hs, ws = h * stride, w * stride
                    for c_in in range(in_channels):
                        for i in range(kernel_size):
                            for j in range(kernel_size):
                                val = 0
                                for c_out in range(out_channel):
                                    val += out_grad[n, c_out, h, w] * weight[c_out, c_in, i, j]
                                in_grad[n, c_in, hs + i, ws + j] += val
        return in_grad

class MaxPoolLayer(object):
    def __init__(self, kernel_size=2, stride=2):
        # 池化核大小
        self.kernel_size = kernel_size
        # 步长
        self.stride = stride
        # 池化索引，用于反向传播
        self.argidx = None
        # 输入张量形状
        self.input_shape = None
        # 输出张量形状
        self.output_shape = None

    def forward(self, inputs):
        # inputs: (N,C,H,W)
        batch, channel, height, width = inputs.shape

        # 获取输出张量的宽高，并构建输出张量
        out_height = int((height - self.kernel_size) / self.stride + 1)
        out_width = int((width - self.kernel_size) / self.stride + 1)
        outputs = np.zeros((batch, channel, out_height, out_width), dtype=inputs.dtype)

        # 记录输入张量和输出张量的形状，并初始化池化索引
        self.input_shape = inputs.shape
        self.output_shape = outputs.shape
        self.argidx = np.zeros_like(outputs, dtype=np.int32)

        # 正向传播
        outputs, self.argidx = self._pool(outputs, inputs, self.argidx, self.kernel_size, self.stride)
        return outputs

    def backward(self, out_grad):
        # 构建输入梯度
        in_grad = np.zeros(self.input_shape)

        # 反向传播
        in_grad = self._pool_back(out_grad, in_grad , self.argidx, self.kernel_size, self.stride)
        return in_grad

    @staticmethod
    @jit(nopython=True)
    def _pool(outputs, inputs, argidx, kernel_size, stride):
        # TODO：根据公式编写下列代码 请用for循环实现
        batch, channel, out_height, out_width = outputs.shape
        for n in range(batch):
            for c in range(channel):
                for h in range(out_height):
                    for w in range(out_width):
                        hs, ws = h*stride, w*stride
                        vector = inputs[n, c, hs:hs+kernel_size, ws:ws+kernel_size]
                        max_value = vector[0][0]
                        for i in range(kernel_size):
                            for j in range(kernel_size):
                                if vector[i, j] > max_value:
                                    max_value = vector[i, j]
                                    # 记录当前索引
                                    argidx[n, c, h, w] = i * kernel_size + j
                        outputs[n, c, h, w] = max_value
        return outputs, argidx

    @staticmethod
    @jit(nopython=True)
    def _pool_back(out_grad, in_grad, argidx, kernel_size, stride):
        # TODO：根据公式编写下列代码 请用for循环实现
        batch, channel, out_height, out_width = out_grad.shape
        for n in range(batch):
            for c in range(channel):
                for h in range(out_height):
                    for w in range(out_width):
                        hs, ws = h*stride, w*stride
                        # 将索引逆向转换至卷积核位置
                        i = argidx[n, c, h, w] // kernel_size
                        j = argidx[n, c, h, w] % kernel_size
                        in_grad[n, c, hs: hs+kernel_size, ws: ws+kernel_size][i, j] = out_grad[n, c, h, w]

        return in_grad

class ReluLayer(object):
    def __init__(self):
        self.argidx = None

    def forward(self, inputs):
        # inputs: (N,C,H,W)
        self.argidx = inputs >= 0
        inputs[inputs < 0] = 0
        return inputs

    def backward(self, out_grad):
        in_grad = out_grad * self.argidx
        return in_grad

class FlattenLayer(object):
    def __init__(self):
        self.input_shape = None

    def forward(self, inputs):
        # inputs: (N,C,H,W) -> (N, C*H*W)
        self.input_shape = inputs.shape
        batch, channel, height, width = inputs.shape
        return inputs.reshape((batch, channel * height * width))

    def backward(self, out_grad):
        return out_grad.reshape(self.input_shape)

class FullyConnectLayer(object):
    def __init__(self, in_features, out_features):
        self.weight = np.random.normal(loc=0, scale=0.01, size=(out_features, in_features))
        self.bias = np.zeros(out_features)

        self.inputs = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, inputs):
        self.inputs = inputs
        bias = np.stack([self.bias for _ in range(inputs.shape[0])])
        return np.dot(inputs, self.weight.T) + bias

    def backward(self, grad):
        self.grad_weight = np.dot(self.inputs.T, grad)
        self.grad_bias = np.matmul(np.ones([grad.shape[0]]), grad)
        grad = np.dot(grad, self.weight)
        return grad

    def load_params(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def update_params(self, lr):
        self.weight = self.weight - lr * self.grad_weight.T
        self.bias = self.bias - lr * self.grad_bias

class CrossEntropy(object):
    def __init__(self, dim=1):
        self.softmax_out = None
        self.label_onehot = None
        self.batch_size = None
        self.dim = dim

    def _softmax(self, inputs, dim=1):
        input_exp = np.exp(inputs)
        partsum = np.sum(input_exp, axis=dim)
        partsum = np.repeat(np.expand_dims(partsum, axis=dim), inputs.shape[dim], axis=dim)
        result = input_exp / partsum

        return result

    def forward(self, inputs, labels):
        self.softmax_out = self._softmax(inputs, dim=self.dim)
        self.batch_size, out_size = self.softmax_out.shape
        self.label_onehot = np.eye(out_size)[labels]
        log_softmax = np.log(self.softmax_out)
        nllloss = -np.sum(self.label_onehot * log_softmax) / labels.shape[0]

        return nllloss

    def backward(self):
        bottom_diff = (self.softmax_out - self.label_onehot) / self.batch_size
        return bottom_diff
