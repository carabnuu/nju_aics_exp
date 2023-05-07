# Copyright 2021 Huawei Technologies Co., Ltd
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
"""postprocess for 310 inference"""
import os
import json
import numpy as np
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy

from model_utils.moxing_adapter import config

if __name__ == '__main__':
    top1_acc = Top1CategoricalAccuracy()
    rst_path = config.result_dir
    if config.dataset_name == "flower_photos":
        labels = np.load(config.label_dir, allow_pickle=True)
        for idx, label in enumerate(labels):
            f_name = os.path.join(rst_path, "VGG16_data_bs" + str(config.batch_size) + "_" + str(idx) + "_0.bin")
            pred = np.fromfile(f_name, np.float32)
            pred = pred.reshape(config.batch_size, int(pred.shape[0] / config.batch_size))
            top1_acc.update(pred, labels[idx])
        print("acc: ", top1_acc.eval())
 
