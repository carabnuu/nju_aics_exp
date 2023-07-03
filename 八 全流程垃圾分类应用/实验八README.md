## 实验八README

### 本实验内容如下：
* 基于Modelarts平台，使用MindSpore框架构建图片分类网络，在垃圾分类训练数据集上完成模型训练，并保存对应的模型权重。

* 基于Modelarts平台，使用MindSpore框架导入训练好的图片分类网络，在垃圾分类测试数据集上完成模型验证和推理。

* 使用AscendCL实现模型应用的部署开发，搭建一个实时的图片分类应用。

### 数据集
请点击数据集链接，下载以下数据集，下载的exp8_garbage.zip解压，解压出 train,test,val 文件夹，和notebook同步目录。

下载链接如下：\
https://openi.pcl.ac.cn/attachments/449d1886-cf6f-4400-bef2-c1fef422652c?type=1

### 环境
- 环境要求
支持Ascend

- 环境准备

| 依赖   | 版本   |
| ------ | ------ |
| c++    | 7.5.0  |
| cmake  | 3.10.2 |
| python | 3.7.5  |
| mindspore | 2.0  |
    
### 设置环境
mindspore安装可以通过MindSpore官网安装页面：https://www.mindspore.cn/install/ ，将MindSpore安装在你的电脑当中。

选择对应参数，直接复制下载链接完成下载
![图片](assets/IMG_1.png)


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

