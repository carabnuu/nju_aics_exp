## 实验五README

### 本实验内容如下：
- 基于CANN高性能算子库构建VGG17网络

- 加载提供的权重（实验四得到的VGG17权重），编译graph得到离线模型，并利用实验三的测试样例对构建的模型进行推理验证，得到推理结果。

### 数据集
请点击数据集链接，下载以下数据集，下载的exp5_infer.zip解压，解压出 ckpt 文件夹和 data 文件夹，和notebook同步目录。

下载链接如下：\
https://openi.pcl.ac.cn/attachments/54bbd211-a19e-43f6-b7c1-a7bf69e98c4c?type=0

### 环境
环境：支持Ascend环境，本实验用的是Ascend310。

版本：python及依赖的库：python3.7.5，c++11.0以上

已完成昇腾AI软件栈在开发环境上的部署（CANN环境，需要完成驱动及CANN软件的安装）。
    
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

