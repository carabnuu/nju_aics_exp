import cv2
import numpy as np


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
    # image -= [125.307, 122.961, 113.8575]
    # image /= [51.5865, 50.847, 51.255]

    # (h,w,c) -> (c,h,w)
    image = image.transpose((2, 0, 1))
    return image


def process_2_bin():
    img_paths = ['./data/daisy_demo.jpg',
                 './data/roses_demo.jpg',
                 './data/sunflowers_demo.jpg',
                 './data/tulips_demo.jpg']
    # 数据预处理
    for img_path in img_paths:
        img = process_image(img_path)
        # 将处理好的图片存为bin格式，以便推理
        img.tofile(img_path.replace('jpg', 'bin'))


if __name__ == "__main__":
    process_2_bin()
