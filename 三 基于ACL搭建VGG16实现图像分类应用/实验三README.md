## 实验三README

### 本实验内容如下：
掌握使用AscendCL（Ascend Computing Language）实现模型应用的开发。将以训练好的`VGG16`网络模型(onnx格式)转换为Davinci架构专用的模型(om格式)，使`VGG16`网络推理过程可以高效的运行在Ascend硬件上。并对指定图片进行推理输出指定结果，搭建一个实时的图片分类应用。

### 数据集
本实验提供已经用atc工具转换好的的`.om`模型文件，并将推理图片进行预处理后送入到 VGG16 模型进行推理。

请点击数据集链接，下载以下数据集，下载的exp3_infer.zip解压，解压出 data 和 model 文件夹，和notebook同步目录，权重文件和测试图片下载链接如下：
https://openi.pcl.ac.cn/attachments/62754804-f05f-4120-93bc-0e77d9c5a62a?type=0

### 环境
1、环境要求：支持Ascend环境(涉及ACL推理，必须使用Ascend芯片）\
2、环境准备：已完成昇腾AI软件栈在开发环境的部署（CANN环境，需要完成驱动及CANN软件的安装，关于CANN环境的安装参考官方文档 https://support.huawei.com/enterprise/zh/doc/EDOC1100164870/e2696354 ）。

本实验依赖的python包如下：
| 依赖          | 版本     |
| ------------- | -------- |
| python        | 3.7.5    |
| c++           | 7.5.0    |
| opencv-pyhton | 4.5.3.56 |
| spicy         | 1.5.4    |
| numpy         | 1.19.4   |
| Make          | 3.10.2   |

### 设置环境
驱动安装及CANN软件的安装参见链接：
 https://support.huawei.com/enterprise/zh/doc/EDOC1100164870/c9443097
- 下载对应操作系统的开发套件包 ，这里以linux系统为例
   Ascend-cann-toolkit_{version}_linux-{arch}.run
   
- 以软件包的安装用户登录安装环境。
- 若安装OS依赖中安装依赖的用户为root用户，则软件包的安装用户可自行指定；若安装OS依赖中安装依赖的用户为非root用户，请确保软件包的安装用户与该用户保持一致。
- 将获取到的开发套件包上传到安装环境任意路径（如“/home/package”）。
- 进入软件包所在路径。
- 增加对软件包的可执行权限。chmod +x *.run
- 执行如下命令校验软件包安装文件的一致性和完整性。
./*.run --check

- 执行以下命令安装软件。root用户默认使用
   ./*.run --install

  - 若用户需自行指定运行用户：
./*.run --install-username=username --install-usergroup=usergroup --install

