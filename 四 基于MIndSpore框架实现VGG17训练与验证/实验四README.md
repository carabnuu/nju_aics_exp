## 实验四README

### 本实验内容如下：
* 熟悉和使用MindSpore框架和ModelArts，熟悉MindSpore常见API的使用方法，熟悉ModelArts一站式模型训练和部署平台。
* 基于MindSpore框架构建VGG17网络。利用花卉数据集上完成模型训练（训练平台：ModelArts，可采用昇腾910芯片进行训练）。模型训练完成后，对模型进行保存。
* 基于昇腾310推理芯片作为计算平台，利用MindSpore框架导入训练好的模型，并在花卉测试数据集对构建的模型进行推理验证，输出推理性能以及测试集正确率。
* 本实验希望借助MindSpore帮助学生熟悉使用深度学习框架，感受框架封装基本操作的便捷。

### 数据集
我们示例中用到的图像花卉数据集，总共包括5种花的类型：分别是daisy（雏菊，633张），dandelion（蒲公英，898张），roses（玫瑰，641张），sunflowers（向日葵，699张），tulips（郁金香，799张），保存在5个文件夹当中，总共3670张，大小大概在230M左右。为了在模型部署上线之后进行测试，数据集在这里分成了 flower_photos_train 和 flower_photos_test 两部分。

请点击数据集链接，下载以下数据集，下载的data.zip保存到code文件夹下，即和notebook同步目录

数据集链接：https://openi.pcl.ac.cn/attachments/88c31019-22cc-41ed-a31c-8f7b11435b60?type=1

### 环境
环境：支持GPU和Ascend环境 \
版本：MindSpore 2.0 & 编程语言：Python 3.7 \
    在动手进行实践之前，确保你已经正确安装了MindSpore。如果没有，可以通过MindSpore官网安装页面：https://www.mindspore.cn/install/ ，将MindSpore安装在你的电脑当中。
    
### 设置环境
选择对应参数，直接复制下载链接完成下载
![图片](assets/IMG_1.png)

