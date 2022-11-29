# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Image classifiation.
"""
import mindspore.nn as nn


class Vgg(nn.Cell):
    """
    VGG网络定义.

    参数:
        num_classes (int): Class numbers. Default: 5.
        phase (int): 指定是训练/评估阶段

    返回值:
        Tensor, infer output tensor.
        
    example：
    	self.layer1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,weight_init='XavierUniform')
        self.layer1_bn1 = nn.BatchNorm2d(num_features=64)
        self.layer1_relu1 = nn.LeakyReLU()

    """
    def __init__(self, num_classes=5, args=None, phase="train"):
        super(Vgg, self).__init__()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        
        self.layer1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,weight_init='XavierUniform')
        self.layer1_bn1 = nn.BatchNorm2d(num_features=64)
        self.layer1_relu1 = nn.ReLU()
        self.layer1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,weight_init='XavierUniform')
        self.layer1_bn2 = nn.BatchNorm2d(num_features=64)
        self.layer1_relu2 = nn.ReLU()
        self.layer1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,weight_init='XavierUniform')
        self.layer2_bn1 = nn.BatchNorm2d(num_features=128)
        self.layer2_relu1 = nn.ReLU()
        self.layer2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,weight_init='XavierUniform')
        self.layer2_bn2 = nn.BatchNorm2d(num_features=128)
        self.layer2_relu2 = nn.ReLU()
        self.layer2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,weight_init='XavierUniform')
        self.layer3_bn1 = nn.BatchNorm2d(num_features=256)
        self.layer3_relu1 = nn.ReLU()
        self.layer3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,weight_init='XavierUniform')
        self.layer3_bn2 = nn.BatchNorm2d(num_features=256)
        self.layer3_relu2 = nn.ReLU()
        self.layer3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,weight_init='XavierUniform')
        self.layer3_bn3 = nn.BatchNorm2d(num_features=256)
        self.layer3_relu3 = nn.ReLU()
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer4_bn1 = nn.BatchNorm2d(num_features=512)
        self.layer4_relu1 = nn.ReLU()
        self.layer4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer4_bn2 = nn.BatchNorm2d(num_features=512)
        self.layer4_relu2 = nn.ReLU()
        self.layer4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer4_bn3 = nn.BatchNorm2d(num_features=512)
        self.layer4_relu3 = nn.ReLU()
        self.layer4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer5_bn1 = nn.BatchNorm2d(num_features=512)
        self.layer5_relu1 = nn.ReLU()
        self.layer5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer5_bn2 = nn.BatchNorm2d(num_features=512)
        self.layer5_relu2 = nn.ReLU()
        self.layer5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer5_bn3 = nn.BatchNorm2d(num_features=512)
        self.layer5_relu3 = nn.ReLU()
        self.layer5_conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,weight_init='XavierUniform')
        self.layer5_bn4 = nn.BatchNorm2d(num_features=512)
        self.layer5_relu4 = nn.ReLU()
        self.layer5_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fullyconnect1 = nn.Dense(512 * 7 * 7, 4096)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_ratio)

        self.fullyconnect2 = nn.Dense(4096, 4096)
        self.relu_2 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout_ratio)

        self.fullyconnect3 = nn.Dense(4096, num_classes)


    def construct(self, x):
        x  =  self.layer1_conv1(x) 
        x  =  self.layer1_bn1(x)
        x  =  self.layer1_relu1(x) 
        x  =  self.layer1_conv2(x)
        x  =  self.layer1_bn2(x)
        x  =  self.layer1_relu2(x) 
        x  =  self.layer1_maxpool(x)

        x  =  self.layer2_conv1(x)
        x  =  self.layer2_bn1(x) 
        x  =  self.layer2_relu1(x) 
        x  =  self.layer2_conv2(x)
        x  =  self.layer2_bn2(x) 
        x  =  self.layer2_relu2(x) 
        x  =  self.layer2_maxpool(x)

        x  =  self.layer3_conv1(x)
        x  =  self.layer3_bn1(x) 
        x  =  self.layer3_relu1(x) 
        x  =  self.layer3_conv2(x)
        x  =  self.layer3_bn2(x) 
        x  =  self.layer3_relu2(x) 
        x  =  self.layer3_conv3(x)
        x  =  self.layer3_bn3(x) 
        x  =  self.layer3_relu3(x) 
        x  =  self.layer3_maxpool(x)

        x  =  self.layer4_conv1(x)
        x  =  self.layer4_bn1(x) 
        x  =  self.layer4_relu1(x) 
        x  =  self.layer4_conv2(x)
        x  =  self.layer4_bn2(x) 
        x  =  self.layer4_relu2(x) 
        x  =  self.layer4_conv3(x)
        x  =  self.layer4_bn3(x)
        x  =  self.layer4_relu3(x) 
        x  =  self.layer4_maxpool(x)
        
        x  =  self.layer5_conv1(x)
        x  =  self.layer5_bn1(x) 
        x  =  self.layer5_relu1(x) 
        x  =  self.layer5_conv2(x)
        x  =  self.layer5_bn2(x) 
        x  =  self.layer5_relu2(x) 
        x  =  self.layer5_conv3(x)
        x  =  self.layer5_bn3(x) 
        x  =  self.layer5_relu3(x) 
        x  =  self.layer5_conv4(x)
        x  =  self.layer5_bn4(x) 
        x  =  self.layer5_relu4(x) 
        x  =  self.layer5_maxpool(x)

        x = self.flatten(x) 
        x = self.fullyconnect1(x) 
        x = self.relu_1(x)
        x = self.dropout_1(x) 
        x = self.fullyconnect2(x)
        x = self.relu_2(x) 
        x = self.dropout_1(x) 
        x = self.fullyconnect3(x) 

        return x



def vgg17(num_classes=1000, args=None, phase="train", **kwargs):
    """
    生成VGG17网络实例 
    参数:
        num_classes (int): 分类数
        args (namespace): 参数
        phase (str): 指定是训练/评估阶段
    返回:
        Cell, cell instance of Vgg17 neural network with Batch Normalization.

    参考如下:
        >>> vgg17(num_classes=5, args=args, **kwargs)
    """
    net = Vgg(num_classes=num_classes, args=args, phase=phase, **kwargs)
    return net
