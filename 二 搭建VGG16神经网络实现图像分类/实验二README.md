## 实验二README

### 本实验内容如下：
本实验实现的是用Python语言构建VGG卷积神经网络，整体流程如下：
- Convolution算子的正向传播及反向传播代码实现
- MaxPool算子的正向传播及反向传播代码实现
- VGG16网络搭建


### 数据集
实验数据集[MNIST](http://yann.lecun.com/exdb/mnist/)，由250个不同的人手写而成，总共有70000张手写数据集。其中训练集有60000张，测试集有10000张。每张图片大小为28×28。
数据集下载链接：
https://openi.pcl.ac.cn/attachments/db2b427e-5a59-464e-b9b5-1aea15301482?type=0

### 环境
环境：支持CPU，GPU和Ascend环境
环境依赖：这是本实验需要的依赖包

| 依赖          | 版本     |
| :-------------: | :--------: |
| python        | 3.7.5    |
| numpy         | 1.19.4   |
| scipy         | 1.5.4    |
| opencv-python | 4.5.3.56 |
| numba         | 0.56.0   |

### 安装
python的相关依赖直接pip install \<package\>即可
