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
"""export checkpoint file into models"""
import os
import numpy as np

from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, export

from src.seresnet import se_resnet50

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    config.image_size = list(map(int, config.image_size.split(',')))

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        config.device_id = get_device_id()
        context.set_context(device_id=config.device_id)

    net = se_resnet50(class_num=config.num_classes)
    

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    input_data = Tensor(np.zeros([config.batch_size, 3, config.image_size[0], config.image_size[1]]), mstype.float32)
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    config.config_path = 'garbage_config.yaml'
    config.ckpt_file = './pretrained/senet50_acc70.ckpt'
    config.file_name = 'garbage_seresnet50'
    config.device_target = 'Ascend'
    config.file_format = 'MINDIR'
    run_export()
