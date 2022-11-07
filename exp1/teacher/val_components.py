import numpy as np
from exp1.teacher.components import FullyConnectLayersrc, ReluLayersrc, CrossEntropysrc
from exp1.student.components import FullyConnectLayer, ReluLayer, CrossEntropy


def compare(src_op, dst_op, inputs, label=None):
    src_out = src_op.forward(inputs, label) if label is not None else src_op.forward(inputs)
    dst_out = dst_op.forward(inputs, label) if label is not None else dst_op.forward(inputs)

    assert np.abs(src_out.mean() - dst_out.mean()) < 1e-3, "wrong %s forward" % src_op.op_name
    assert np.abs(src_out.sum() - dst_out.sum()) < 1e-3, "wrong %s forward" % src_op.op_name

    loss = src_out
    src_grad = src_op.backward(loss)
    dst_grad = dst_op.backward(loss)

    assert np.abs(src_grad.mean() - dst_grad.mean()) < 1e-3, "wrong %s backward" % src_op.op_name
    assert np.abs(src_grad.sum() - dst_grad.sum()) < 1e-3, "wrong %s backward" % src_op.op_name


if __name__ == '__main__':
    # test fc
    for _ in range(10):
        inputs = np.random.randn(4, 10)
        src_op = FullyConnectLayersrc(in_features=10, out_features=16)
        dst_op = FullyConnectLayer(in_features=10, out_features=16)
        dst_op.load_params(src_op.weight, src_op.bias)
        compare(src_op, dst_op, inputs)
    print('component FC pass')

    # test relu
    for _ in range(10):
        inputs = np.random.randn(4, 3, 10, 10)
        src_op = ReluLayersrc()
        dst_op = ReluLayer()
        compare(src_op, dst_op, inputs)
    print('component ReLU pass')

    # test CrossEntropy
    for _ in range(10):
        inputs = np.random.randn(16, 10)
        labels = np.random.randint(0, 10, 16)
        src_op = CrossEntropysrc()
        dst_op = CrossEntropy()
        compare(src_op, dst_op, inputs, labels)
    print('component CrossEntropy pass')
