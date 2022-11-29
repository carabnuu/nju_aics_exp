import numpy as np


class FullyConnectLayersrc(object):
    def __init__(self, in_features, out_features, has_bias=True):
        # 初始化权重和偏置
        self.weight = np.random.normal(loc=0, scale=0.01, size=(out_features, in_features))
        self.bias = np.zeros(out_features) if has_bias else None
        self.has_bias = has_bias

        self.inputs = None
        self.grad_weight = None
        self.grad_bias = None
        self.op_name = 'FCLayer'

    def forward(self, inputs):
        # 根据公式编写全连接层的前向传播过程
        self.inputs = inputs
        bias = np.stack([self.bias for _ in range(inputs.shape[0])]) if self.has_bias else 0
        outputs = np.dot(inputs, self.weight.T) + bias
        return outputs

    def backward(self, in_grad):
        # 根据公式编写全连接层的反向传播过程
        self.grad_weight = np.dot(self.inputs.T, in_grad)
        self.grad_bias = np.matmul(np.ones([in_grad.shape[0]]), in_grad)
        out_grad = np.dot(in_grad, self.weight)
        return out_grad

    def update_params(self, lr):
        # 根据公式编写全连接层的参数更新过程
        self.weight = self.weight - lr * self.grad_weight.T
        if self.has_bias:
            self.bias = self.bias - lr * self.grad_bias

    def load_params(self, weight, bias):
        # 加载权重和偏置
        assert self.weight.shape == weight.shape
        self.weight = weight
        if self.has_bias:
            assert self.bias.shape == bias.shape
            self.bias = bias


class ReluLayersrc(object):
    def __init__(self):
        self.inputs = None
        self.op_name = 'ReLULayer'

    def forward(self, inputs):
        # 根据公式编写激活函数ReLU的前向传播过程
        self.inputs = inputs
        outputs = np.maximum(self.inputs, 0)
        return outputs

    def backward(self, in_grad):
        # 根据公式编写激活函数ReLU的反向传播过程
        b = self.inputs
        b[b > 0] = 1
        b[b < 0] = 0
        out_grad = np.multiply(b, in_grad)
        return out_grad


class CrossEntropysrc(object):
    def __init__(self, dim=1):
        self.softmax_out = None
        self.label_onehot = None
        self.batch_size = None
        self.dim = dim
        self.op_name = 'CrossEntropy'

    def _softmax(self, inputs, dim=1):
        input_exp = np.exp(inputs)
        partsum = np.sum(input_exp, axis=dim)
        partsum = np.repeat(np.expand_dims(partsum, axis=dim), inputs.shape[dim], axis=dim)
        result = input_exp / partsum

        return result

    def forward(self, inputs, labels):
        # 根据公式编写交叉熵损失函数的前向传播过程
        self.softmax_out = self._softmax(inputs, dim=self.dim)
        self.batch_size, out_size = self.softmax_out.shape
        self.label_onehot = np.eye(out_size)[labels]
        log_softmax = np.log(self.softmax_out)
        outputs = -np.sum(self.label_onehot * log_softmax) / labels.shape[0]

        return outputs

    def backward(self, in_grad):
        # 根据公式编写交叉熵损失函数的反向传播过程
        out_grad = (self.softmax_out - self.label_onehot) / self.batch_size

        return out_grad
