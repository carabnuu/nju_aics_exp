# -*- coding: UTF-8 -*-
import struct
import time
import acl
from process import process_image
import numpy as np


CLASSES = ("daisy", "roses", "sunflowers", "tulips")
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
NPY_BYTE = 1


class ACLEngine(object):
    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.model_desc = None
        self.load_input_dataset, self.load_output_dataset = None, None
        self.input_data, self.output_data = None, None

        # --------------- pyACL初始化 ---------------
        print('Start init resource')
        self._init_resource()
        self._load_model(model_path)
        self._prepare_inputs()

    def inference(self, img_path):
        """ 模型推理及后处理模块 """
        # TODO 根据操作手册完善该函数
        # 1.读取并预处理图片
        # 2.准备模型推理的输入数据，运行模式默认为运行模式为ACL_HOST，当前实例代码中模型只有一个输入。
        # 将图片数据从Host传输到Device。

        # 3.执行模型推理。
        # self.model_id表示模型ID，在模型加载成功后，会返回标识模型的ID。

        # 4.处理模型推理的输出数据，输出置信度的类别编号。
        pass

    def release_resource(self):
        """ 资源释放模块 """
        self._unload_model()
        self._unload_picture()
        self._destroy_resource()
        print('Resource destroyed successfully')

    def _init_resource(self):
        # TODO 根据操作手册完善该函数
        # pyACL初始化
        # 运行管理资源申请
        # 指定运算的Device。
        # 显式创建一个Context，用于管理Stream对象。
        pass

    def _load_model(self, model_path):
        # TODO 根据操作手册完善该函数
        # 加载离线模型文件，返回标识模型的ID。
        # 根据加载成功的模型的ID，获取该模型的描述信息。
        pass

    def _prepare_inputs(self):
        # TODO 根据操作手册完善该函数
        # 1.准备模型推理的输入数据集。
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        # 获取模型输入的数量。
        # 循环为每个输入申请内存，并将每个输入添加到aclmdlDataset类型的数据中。

        # 2.准备模型推理的输出数据集。
        # 创建aclmdlDataset类型的数据，描述模型推理的输出。
        # 获取模型输出的数量。
        # 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中。
        pass

    def _unload_model(self):
        # TODO 根据操作手册完善该函数
        # 卸载模型。
        # 释放模型描述信息。
        # 释放Context。
        pass

    def _unload_picture(self):
        # TODO 根据操作手册完善该函数
        # 释放输出资源，包括数据结构和内存。
        pass

    def _destroy_resource(self):
        # TODO 根据操作手册完善该函数
        # 释放Device。
        # pyACL去初始化。
        pass


if __name__ == '__main__':
    engine = ACLEngine('./model/vgg16.om')
    # engine = ACLEngine('./model/vgg16_fp32.om')
    engine.inference('./data/daisy_demo.jpg')
    engine.inference('./data/roses_demo.jpg')
    engine.inference('./data/sunflowers_demo.jpg')
    engine.inference('./data/tulips_demo.jpg')
    engine.release_resource()
