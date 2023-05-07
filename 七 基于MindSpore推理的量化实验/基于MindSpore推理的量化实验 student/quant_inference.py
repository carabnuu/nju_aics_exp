import numpy as np
import cv2
import mindspore as ms
from mindspore import ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from vgg import Vgg
context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
np.set_printoptions(suppress=True)


def resize_image(image, target_size):
    h, w = image.shape[:2]
    th, tw = target_size
    # 获取等比缩放后的尺寸
    scale = min(th / h, tw / w)
    oh, ow = round(h * scale), round(w * scale)
    # 缩放图片，opencv缩放传入尺寸为（宽，高），这里采用线性差值算法
    image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    # 将剩余部分进行填充
    new_image = np.ones((th, tw, 3), dtype=np.uint8) * 114
    new_image[:oh, :ow, :] = image
    return new_image


def process_image(img_path):
    # 读取图片，opencv读图后格式是BGR格式，需要转为RGB格式
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图片等比resize至(224x224)
    image = resize_image(image, (224, 224))
    image = np.array(image, dtype=np.float32)
    # 将图片标准化
    image -= [125.307, 122.961, 113.8575]
    image /= [51.5865, 50.847, 51.255]
    # (h,w,c) -> (c,h,w) -> (1,c,h,w)
    image = image.transpose((2, 0, 1))[None]
    return image


def direct_quantize(model, dataset):
    print('*'*50)
    print('Start quantize')
    for img_path, label in dataset:
        print("Start inference: {}".format(img_path))
        ndarray = process_image(img_path)
        tensor = ms.Tensor(ndarray, ms.float32)
        net_out = model.quantize_forward(tensor)
        prob = ops.Softmax()(net_out)
        print('Predict probability: {}'.format(np.around(prob.asnumpy(), 4)))
        predict_cls = (ops.Argmax()(prob)).asnumpy().item()
        print('Inference result: {}\n'.format(predict_cls == label))


def full_inference(model, dataset):
    print('*' * 50)
    print('Start full inference')
    for img_path, label in dataset:
        print("Start inference: {}".format(img_path))
        ndarray = process_image(img_path)
        tensor = ms.Tensor(ndarray, ms.float32)
        net_out = model(tensor)
        prob = ops.Softmax()(net_out)
        print('Predict probability: {}'.format(np.around(prob.asnumpy(), 4)))
        predict_cls = (ops.Argmax()(prob)).asnumpy().item()
        print('Inference result: {}\n'.format(predict_cls == label))


def quantize_inference(model, dataset):
    print('*' * 50)
    print('Start quantize inference')
    for img_path, label in dataset:
        print("Start inference: {}".format(img_path))
        ndarray = process_image(img_path)
        tensor = ms.Tensor(ndarray, ms.float32)
        net_out = model.quantize_inference(tensor)
        prob = ops.Softmax()(net_out)
        print('Predict probability: {}'.format(np.around(prob.asnumpy(), 4)))
        predict_cls = (ops.Argmax()(prob)).asnumpy().item()
        print('Inference result: {}\n'.format(predict_cls == label))


if __name__ == '__main__':
    # 初始化VGG网络并加载权重系数
    net = Vgg(num_classes=4)
    load_param_into_net(net, load_checkpoint('vgg.ckpt'), strict_load=True)
    net.set_train(False)

    # 构建对应推理数据
    dataset = [('./data/daisy_demo.jpg', 0),
               ('./data/roses_demo.jpg', 1),
               ('./data/sunflowers_demo.jpg', 2),
               ('./data/tulips_demo.jpg', 3)]

    # 首先进行正常的网络推理，获取模型输出
    full_inference(net, dataset)

    # 构建量化模型，此实验为int8量化
    net.quantize(num_bits=8)

    # 进行量化推理，这里涉及到对中间特征图统计最大最小值
    direct_quantize(net, dataset)

    # 对网络量化参数进行固定
    net.freeze()

    # 进行量化推理
    quantize_inference(net, dataset)
