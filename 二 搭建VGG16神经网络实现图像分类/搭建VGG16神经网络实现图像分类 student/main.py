import cv2
import numpy as np
from vgg import VGG16
from layer import CrossEntropy


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


if __name__ == '__main__':
    import time
    # 分类类别
    CLASSES = ('daisy', 'roses', 'sunflowers', 'tulips')

    # 网络初始化、加载权重参数
    model = VGG16(4)
    ckpt = np.load('./file/vgg16_ckpt.npy', allow_pickle=True).item()
    model.resume_weights(ckpt)

    start_time = time.time()

    # 输入图片预处理
    image_path = './file/tulips_demo.jpg'
    tensor = process_image(image_path)

    # 模型正向传播
    outputs = model.forward(tensor)
    print(outputs)
    pred = int(np.argmax(outputs))
    print(CLASSES[pred])

    # 计算loss
    label = np.array([1, ])
    loss_func = CrossEntropy()
    loss = loss_func.forward(outputs, label)
    print(loss)

    # 反向传播
    grad = loss_func.backward()
    grad = model.backward(grad)
    print(grad.mean(), grad.sum())

    end_time = time.time()
    print(end_time - start_time)

