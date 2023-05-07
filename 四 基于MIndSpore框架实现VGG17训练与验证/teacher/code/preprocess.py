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
"""preprocess"""
import os
import json
import numpy as np
from src.dataset import vgg_create_dataset

from model_utils.moxing_adapter import config

 

if __name__ == "__main__":
    if config.dataset == "flower_photos":
        dataset = vgg_create_dataset(config.data_dir, config.image_size, config.per_batch_size, training=False)
        img_path = os.path.join(config.result_path, "00_data")
        os.makedirs(img_path)
        label_list = []
        for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
            file_name = "VGG17_data_bs" + str(config.per_batch_size) + "_" + str(idx) + ".bin"
            file_path = os.path.join(img_path, file_name)
            data["image"].tofile(file_path)
            label_list.append(data["label"])
        np.save(os.path.join(config.result_path, "flower_label_ids.npy"), label_list)
        print("=" * 20, "export bin files finished", "=" * 20)
 