import cv2
import numpy as np
import struct
import time
import acl

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
NPY_BYTE = 1


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

    # (h,w,c) -> (c,h,w)
    image = image.transpose((2, 0, 1))
    return image


def softmax(f):
    return np.exp(f) / np.sum(np.exp(f))

class ACLEngine(object):
    def __init__(self, model_path, class_map, device_id=0):
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.model_desc = None
        self.load_input_dataset, self.load_output_dataset = None, None
        self.input_data, self.output_data = None, None
        self.class_map = class_map
        self.num_classes = len(class_map)

        # --------------- pyACL初始化 ---------------
        print('Start init resource')
        self._init_resource()
        self._load_model(model_path)
        self._prepare_inputs()

    def inference(self, img_path):
        """ 模型推理及后处理模块 """
        # 1.读取并预处理图片
        img = process_image(img_path)
        # 2.准备模型推理的输入数据，运行模式默认为运行模式为ACL_HOST，当前实例代码中模型只有一个输入。

        bytes_data = img.tobytes()
        np_ptr = acl.util.bytes_to_ptr(bytes_data)

        start_time = time.time()
        # 将图片数据从Host传输到Device。
        ret = acl.rt.memcpy(self.input_data[0]["buffer"], self.input_data[0]["size"], np_ptr,
                            self.input_data[0]["size"], ACL_MEMCPY_HOST_TO_DEVICE)

        # 3.执行模型推理。
        # self.model_id表示模型ID，在模型加载成功后，会返回标识模型的ID。
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)

        # 4.处理模型推理的输出数据，输出置信度的类别编号。
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将推理输出数据从Device传输到Host。
            ret = acl.rt.memcpy(buffer_host, self.output_data[i]["size"], self.output_data[i]["buffer"],
                                self.output_data[i]["size"], ACL_MEMCPY_DEVICE_TO_HOST)

            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            data = np.frombuffer(bytes_out, dtype=np.byte)
            inference_result.append(data)

        tuple_st = struct.unpack(f"{self.num_classes}f", bytearray(inference_result[0]))
        vals = np.array(tuple_st).flatten()
        vals = softmax(vals)
        top_k = vals.argsort()[-1:-6:-1]
        print("\n======== inference results: =============")
        for i, j in enumerate(top_k):
            print("top %d: class:[%s]: probability:[%f]" % (i, self.class_map[str(j)], vals[j]))

        end_time = time.time()
        print('inference cost time: {:.1f}ms\n'.format((end_time-start_time)*1000))

    def release_resource(self):
        """ 资源释放模块 """
        self._unload_model()
        self._unload_picture()
        self._destroy_resource()
        print('Resource destroyed successfully')

    def _init_resource(self):
        # pyACL初始化
        ret = acl.init()

        # 运行管理资源申请
        # 指定运算的Device。
        self.device_id = 0
        ret = acl.rt.set_device(self.device_id)
        # 显式创建一个Context，用于管理Stream对象。
        self.context, ret = acl.rt.create_context(self.device_id)

    def _load_model(self, model_path):
        # 加载离线模型文件，返回标识模型的ID。
        self.model_id, ret = acl.mdl.load_from_file(model_path)

        # 根据加载成功的模型的ID，获取该模型的描述信息。
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)

    def _prepare_inputs(self):
        # 1.准备模型推理的输入数据集。
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        self.load_input_dataset = acl.mdl.create_dataset()
        # 获取模型输入的数量。
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.input_data = []
        # 循环为每个输入申请内存，并将每个输入添加到aclmdlDataset类型的数据中。
        for i in range(input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            # 申请输入内存。
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_input_dataset, data)
            self.input_data.append({"buffer": buffer, "size": buffer_size})

        # 2.准备模型推理的输出数据集。
        # 创建aclmdlDataset类型的数据，描述模型推理的输出。
        self.load_output_dataset = acl.mdl.create_dataset()
        # 获取模型输出的数量。
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.output_data = []
        # 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中。
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            # 申请输出内存。
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            self.output_data.append({"buffer": buffer, "size": buffer_size})

    def _unload_model(self):
        # 卸载模型。
        ret = acl.mdl.unload(self.model_id)
        # 释放模型描述信息。
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        # 释放Context。
        if self.context:
            ret = acl.rt.destroy_context(self.context)
            self.context = None

    def _unload_picture(self):
        # 释放输出资源，包括数据结构和内存。
        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
        output_number = acl.mdl.get_dataset_num_buffers(self.load_output_dataset)
        for i in range(output_number):
            data_buf = acl.mdl.get_dataset_buffer(self.load_output_dataset, i)
            if data_buf:
                ret = acl.destroy_data_buffer(data_buf)
        ret = acl.mdl.destroy_dataset(self.load_output_dataset)

    def _destroy_resource(self):
        # 释放Device。
        ret = acl.rt.reset_device(self.device_id)
        # pyACL去初始化。
        ret = acl.finalize()


if __name__ == '__main__':
    import json

    deploy_model_filename = "deploy_model/garbage_deploy"
    label_list_json_path = "../data/label_list.json"
    device_id = 0

    with open(label_list_json_path, 'r', encoding='utf-8') as json_path:
        class_map = json.load(json_path)

    # 初始化推理引擎
    engine = ACLEngine(f'{deploy_model_filename}.om', class_map, device_id=device_id)
    # 推理单张图片
    engine.inference('../data/test/1/img_352.jpg')
    # 释放资源
    engine.release_resource()