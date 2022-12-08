import mindspore.nn as nn
from quant_module import QConv2d, QMaxPooling2d, QDense, QReLU


class Vgg(nn.Cell):
    def __init__(self, num_classes=4):
        super(Vgg, self).__init__()
        self.layer1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, has_bias=True)
        self.layer1_relu1 = nn.ReLU()
        self.layer1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, has_bias=True)
        self.layer1_relu2 = nn.ReLU()
        self.layer1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, has_bias=True)
        self.layer2_relu1 = nn.ReLU()
        self.layer2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, has_bias=True)
        self.layer2_relu2 = nn.ReLU()
        self.layer2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, has_bias=True)
        self.layer3_relu1 = nn.ReLU()
        self.layer3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, has_bias=True)
        self.layer3_relu2 = nn.ReLU()
        self.layer3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, has_bias=True)
        self.layer3_relu3 = nn.ReLU()
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, has_bias=True)
        self.layer4_relu1 = nn.ReLU()
        self.layer4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, has_bias=True)
        self.layer4_relu2 = nn.ReLU()
        self.layer4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, has_bias=True)
        self.layer4_relu3 = nn.ReLU()
        self.layer4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, has_bias=True)
        self.layer5_relu1 = nn.ReLU()
        self.layer5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, has_bias=True)
        self.layer5_relu2 = nn.ReLU()
        self.layer5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, has_bias=True)
        self.layer5_relu3 = nn.ReLU()
        self.layer5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fullyconnect1 = nn.Dense(512 * 7 * 7, 4096)
        self.relu_1 = nn.ReLU()
        self.fullyconnect2 = nn.Dense(4096, 4096)
        self.relu_2 = nn.ReLU()
        self.fullyconnect3 = nn.Dense(4096, num_classes)

    def construct(self, x):
        x = self.layer1_conv1(x)
        x = self.layer1_relu1(x)
        x = self.layer1_conv2(x)
        x = self.layer1_relu2(x)
        x = self.layer1_maxpool(x)

        x = self.layer2_conv1(x)
        x = self.layer2_relu1(x)
        x = self.layer2_conv2(x)
        x = self.layer2_relu2(x)
        x = self.layer2_maxpool(x)

        x = self.layer3_conv1(x)
        x = self.layer3_relu1(x)
        x = self.layer3_conv2(x)
        x = self.layer3_relu2(x)
        x = self.layer3_conv3(x)
        x = self.layer3_relu3(x)
        x = self.layer3_maxpool(x)

        x = self.layer4_conv1(x)
        x = self.layer4_relu1(x)
        x = self.layer4_conv2(x)
        x = self.layer4_relu2(x)
        x = self.layer4_conv3(x)
        x = self.layer4_relu3(x)
        x = self.layer4_maxpool(x)

        x = self.layer5_conv1(x)
        x = self.layer5_relu1(x)
        x = self.layer5_conv2(x)
        x = self.layer5_relu2(x)
        x = self.layer5_conv3(x)
        x = self.layer5_relu3(x)
        x = self.layer5_maxpool(x)

        x = self.flatten(x)
        x = self.fullyconnect1(x)
        x = self.relu_1(x)
        x = self.fullyconnect2(x)
        x = self.relu_2(x)
        x = self.fullyconnect3(x)
        return x

    def quantize(self, num_bits=8):
        # 第一个卷积模块需要获取量化输入，其余模块会复用之前的量化输出
        self.qlayer1_conv1 = QConv2d(self.layer1_conv1, qi=True, qo=True, num_bits=num_bits)
        self.qlayer1_relu1 = QReLU()
        self.qlayer1_conv2 = QConv2d(self.layer1_conv2, qi=False, qo=True, num_bits=num_bits)
        self.qlayer1_relu2 = QReLU()
        self.qlayer1_maxpool2d = QMaxPooling2d(self.layer1_maxpool)

        self.qlayer2_conv1 = QConv2d(self.layer2_conv1, qi=False, qo=True, num_bits=num_bits)
        self.qlayer2_relu1 = QReLU()
        self.qlayer2_conv2 = QConv2d(self.layer2_conv2, qi=False, qo=True, num_bits=num_bits)
        self.qlayer2_relu2 = QReLU()
        self.qlayer2_maxpool2d = QMaxPooling2d(self.layer2_maxpool)

        self.qlayer3_conv1 = QConv2d(self.layer3_conv1, qi=False, qo=True, num_bits=num_bits)
        self.qlayer3_relu1 = QReLU()
        self.qlayer3_conv2 = QConv2d(self.layer3_conv2, qi=False, qo=True,num_bits=num_bits)
        self.qlayer3_relu2 = QReLU()
        self.qlayer3_conv3 = QConv2d(self.layer3_conv3, qi=False, qo=True, num_bits=num_bits)
        self.qlayer3_relu3 = QReLU()
        self.qlayer3_maxpool2d = QMaxPooling2d(self.layer3_maxpool)

        self.qlayer4_conv1 = QConv2d(self.layer4_conv1, qi=False, qo=True, num_bits=num_bits)
        self.qlayer4_relu1 = QReLU()
        self.qlayer4_conv2 = QConv2d(self.layer4_conv2, qi=False, qo=True, num_bits=num_bits)
        self.qlayer4_relu2 = QReLU()
        self.qlayer4_conv3 = QConv2d(self.layer4_conv3, qi=False, qo=True, num_bits=num_bits)
        self.qlayer4_relu3 = QReLU()
        self.qlayer4_maxpool2d = QMaxPooling2d(self.layer4_maxpool)

        self.qlayer5_conv1 = QConv2d(self.layer5_conv1, qi=False, qo=True, num_bits=num_bits)
        self.qlayer5_relu1 = QReLU()
        self.qlayer5_conv2 = QConv2d(self.layer5_conv2, qi=False, qo=True, num_bits=num_bits)
        self.qlayer5_relu2 = QReLU()
        self.qlayer5_conv3 = QConv2d(self.layer5_conv3, qi=False, qo=True, num_bits=num_bits)
        self.qlayer5_relu3 = QReLU()
        self.qlayer5_maxpool2d = QMaxPooling2d(self.layer5_maxpool)

        self.qfc1 = QDense(self.fullyconnect1, qi=False, qo=True, num_bits=num_bits)
        self.qfc1_relu = QReLU()
        self.qfc2 = QDense(self.fullyconnect2, qi=False, qo=True, num_bits=num_bits)
        self.qfc2_relu = QReLU()
        self.qfc3 = QDense(self.fullyconnect3, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qlayer1_conv1(x)
        x = self.qlayer1_relu1(x)
        x = self.qlayer1_conv2(x)
        x = self.qlayer1_relu2(x)
        x = self.qlayer1_maxpool2d(x)

        x = self.qlayer2_conv1(x)
        x = self.qlayer2_relu1(x)
        x = self.qlayer2_conv2(x)
        x = self.qlayer2_relu2(x)
        x = self.qlayer2_maxpool2d(x)

        x = self.qlayer3_conv1(x)
        x = self.qlayer3_relu1(x)
        x = self.qlayer3_conv2(x)
        x = self.qlayer3_relu2(x)
        x = self.qlayer3_conv3(x)
        x = self.qlayer3_relu3(x)
        x = self.qlayer3_maxpool2d(x)

        x = self.qlayer4_conv1(x)
        x = self.qlayer4_relu1(x)
        x = self.qlayer4_conv2(x)
        x = self.qlayer4_relu2(x)
        x = self.qlayer4_conv3(x)
        x = self.qlayer4_relu3(x)
        x = self.qlayer4_maxpool2d(x)

        x = self.qlayer5_conv1(x)
        x = self.qlayer5_relu1(x)
        x = self.qlayer5_conv2(x)
        x = self.qlayer5_relu2(x)
        x = self.qlayer5_conv3(x)
        x = self.qlayer5_relu3(x)
        x = self.qlayer5_maxpool2d(x)

        x = self.flatten(x)
        x = self.qfc1(x)
        x = self.qfc1_relu(x)
        x = self.qfc2(x)
        x = self.qfc2_relu(x)
        x = self.qfc3(x)
        return x

    def freeze(self):
        # 冻结网络参数时，除第一个卷积模块不需要指定量化输入，其余模块都需要指定量化输入
        self.qlayer1_conv1.freeze()
        self.qlayer1_relu1.freeze(qi=self.qlayer1_conv1.qo)
        self.qlayer1_conv2.freeze(qi=self.qlayer1_conv1.qo)
        self.qlayer1_relu2.freeze(qi=self.qlayer1_conv2.qo)
        self.qlayer1_maxpool2d.freeze(qi=self.qlayer1_conv2.qo)

        self.qlayer2_conv1.freeze(qi=self.qlayer1_conv2.qo)
        self.qlayer2_relu1.freeze(qi=self.qlayer2_conv1.qo)
        self.qlayer2_conv2.freeze(qi=self.qlayer2_conv1.qo)
        self.qlayer2_relu2.freeze(qi=self.qlayer2_conv2.qo)
        self.qlayer2_maxpool2d.freeze(qi=self.qlayer2_conv2.qo)

        self.qlayer3_conv1.freeze(qi=self.qlayer2_conv2.qo)
        self.qlayer3_relu1.freeze(qi=self.qlayer3_conv1.qo)
        self.qlayer3_conv2.freeze(qi=self.qlayer3_conv1.qo)
        self.qlayer3_relu2.freeze(qi=self.qlayer3_conv2.qo)
        self.qlayer3_conv3.freeze(qi=self.qlayer3_conv2.qo)
        self.qlayer3_relu3.freeze(qi=self.qlayer3_conv3.qo)
        self.qlayer3_maxpool2d.freeze(qi=self.qlayer3_conv3.qo)

        self.qlayer4_conv1.freeze(qi=self.qlayer3_conv3.qo)
        self.qlayer4_relu1.freeze(qi=self.qlayer4_conv1.qo)
        self.qlayer4_conv2.freeze(qi=self.qlayer4_conv1.qo)
        self.qlayer4_relu2.freeze(qi=self.qlayer4_conv2.qo)
        self.qlayer4_conv3.freeze(qi=self.qlayer4_conv2.qo)
        self.qlayer4_relu3.freeze(qi=self.qlayer4_conv3.qo)
        self.qlayer4_maxpool2d.freeze(qi=self.qlayer4_conv3.qo)

        self.qlayer5_conv1.freeze(qi=self.qlayer4_conv3.qo)
        self.qlayer5_relu1.freeze(qi=self.qlayer5_conv1.qo)
        self.qlayer5_conv2.freeze(qi=self.qlayer5_conv1.qo)
        self.qlayer5_relu2.freeze(qi=self.qlayer5_conv2.qo)
        self.qlayer5_conv3.freeze(qi=self.qlayer5_conv2.qo)
        self.qlayer5_relu3.freeze(qi=self.qlayer5_conv3.qo)
        self.qlayer5_maxpool2d.freeze(qi=self.qlayer5_conv3.qo)

        self.qfc1.freeze(qi=self.qlayer5_conv3.qo)
        self.qfc1_relu.freeze(qi=self.qfc1.qo)
        self.qfc2.freeze(qi=self.qfc1.qo)
        self.qfc2_relu.freeze(qi=self.qfc2.qo)
        self.qfc3.freeze(qi=self.qfc2.qo)

    def quantize_inference(self, x):
        # 对输入x进行量化
        qx = self.qlayer1_conv1.qi.quantize_tensor(x)

        qx = self.qlayer1_conv1.quantize_inference(qx)
        qx = self.qlayer1_relu1.quantize_inference(qx)
        qx = self.qlayer1_conv2.quantize_inference(qx)
        qx = self.qlayer1_relu2.quantize_inference(qx)
        qx = self.qlayer1_maxpool2d.quantize_inference(qx)

        qx = self.qlayer2_conv1.quantize_inference(qx)
        qx = self.qlayer2_relu1.quantize_inference(qx)
        qx = self.qlayer2_conv2.quantize_inference(qx)
        qx = self.qlayer2_relu2.quantize_inference(qx)
        qx = self.qlayer2_maxpool2d.quantize_inference(qx)

        qx = self.qlayer3_conv1.quantize_inference(qx)
        qx = self.qlayer3_relu1.quantize_inference(qx)
        qx = self.qlayer3_conv2.quantize_inference(qx)
        qx = self.qlayer3_relu2.quantize_inference(qx)
        qx = self.qlayer3_conv3.quantize_inference(qx)
        qx = self.qlayer3_relu3.quantize_inference(qx)
        qx = self.qlayer3_maxpool2d.quantize_inference(qx)

        qx = self.qlayer4_conv1.quantize_inference(qx)
        qx = self.qlayer4_relu1.quantize_inference(qx)
        qx = self.qlayer4_conv2.quantize_inference(qx)
        qx = self.qlayer4_relu2.quantize_inference(qx)
        qx = self.qlayer4_conv3.quantize_inference(qx)
        qx = self.qlayer4_relu3.quantize_inference(qx)
        qx = self.qlayer4_maxpool2d.quantize_inference(qx)

        qx = self.qlayer5_conv1.quantize_inference(qx)
        qx = self.qlayer5_relu1.quantize_inference(qx)
        qx = self.qlayer5_conv2.quantize_inference(qx)
        qx = self.qlayer5_relu2.quantize_inference(qx)
        qx = self.qlayer5_conv3.quantize_inference(qx)
        qx = self.qlayer5_relu3.quantize_inference(qx)
        qx = self.qlayer5_maxpool2d.quantize_inference(qx)

        qx = self.flatten(qx)

        qx = self.qfc1.quantize_inference(qx)
        qx = self.qfc1_relu.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        qx = self.qfc2_relu.quantize_inference(qx)
        qx = self.qfc3.quantize_inference(qx)

        # 对输出qx进行反量化
        out = self.qfc3.qo.dequantize_tensor(qx)
        return out
