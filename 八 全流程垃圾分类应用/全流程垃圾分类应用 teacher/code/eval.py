
"""Eval"""
import os
import time
import datetime
import random 
import mindspore
import glob
import numpy as np
import mindspore.nn as nn

from mindspore import Tensor, context
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
import mindspore.ops as P
from mindspore import dtype as mstype 
from mindspore.train import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor,Callback

from src.utils.logging import get_logger
from src.seresnet import se_resnet50
from src.dataset import resnet_create_dataset
from src.dataset import classification_dataset

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

class ParameterReduce(nn.Cell):
    """ParameterReduce"""
    def __init__(self):
        super(ParameterReduce, self).__init__()
        self.cast = P.Cast()
        self.reduce = P.AllReduce()

    def construct(self, x):
        one = self.cast(F.scalar_to_array(1.0), mstype.float32)
        out = x * one
        ret = self.reduce(out)
        return ret


def get_top5_acc(top5_arg, gt_class):
    sub_count = 0
    for top5, gt in zip(top5_arg, gt_class):
        if gt in top5:
            sub_count += 1
    return sub_count


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        # Each server contains 8 devices as most.
        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """run eval"""
    config.per_batch_size = config.batch_size
    config.image_size = list(map(int, config.image_size.split(',')))
    config.rank = get_rank_id()
    config.group_size = get_device_num()


    _enable_graph_kernel = config.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=False,
                        device_target=config.device_target, save_graphs=False)
    if os.getenv('DEVICE_ID', "not_set").isdigit() and config.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    config.outputs_dir = os.path.join(config.log_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    if config.dataset == "garbage_photos":
        net = se_resnet50(class_num=config.num_classes,phase="test")
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(net, loss_fn=loss, metrics={'acc'})
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict)
        net.set_train(False)
        dataset = resnet_create_dataset(config.data_dir, config.image_size, config.per_batch_size, training=False, valid=True)
        res = model.eval(dataset)
        print("result: ", res)


if __name__ == "__main__":
    config.pre_trained="pretrained/0-20_277.ckpt"
    config.config_path="garbage_config.yaml"

    config.data_dir="../data"
    config.dataset="garbage_photos"
   
    run_eval()
