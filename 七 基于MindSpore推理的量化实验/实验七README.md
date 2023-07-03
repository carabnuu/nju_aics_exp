## 实验七README

### 本实验内容如下：
- 了解神经网络量化操作，能够独立实现int8量化操作
- 构建量化VGG16神经网络
- 基于MindSpore框架实现量化推理，能够独立编写量化操作代码

### 数据集
请点击数据集链接，下载vgg.ckpt模型文件，下载的exp7_data.zip解压，解压重命名为 data 文件夹，和notebook同步目录。

测试图片下载链接如下：https://openi.pcl.ac.cn/attachments/f6076d5f-2b91-4339-93c2-33f38b17821c?type=1

### 环境
环境：支持Ascend

版本：
| 依赖   | 版本   |
| ------ | ------ |
| opencv-pyhton    | 4.5.3.56  |
| mindspore  | 2.0 |
| numpy         | 1.19.4   |
| python | 3.7.5  |
    
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

