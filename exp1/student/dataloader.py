import os
import random
import numpy as np


class DataLoader(object):
    def __init__(self, mnist_npy_dir, batch_size=16, mode='train'):
        self.mode = mode
        self.mnist_npy_dir = mnist_npy_dir
        self.input_data, self.input_label = None, None
        self.prepare_data()
        self.batch_size = batch_size
        self.batch_nums = len(self.input_data) // batch_size

    def shuffle_data(self):
        tmp = list(zip(self.input_data, self.input_label))
        random.shuffle(tmp)
        self.input_data, self.input_label = zip(*tmp)

    def prepare_data(self):
        images = np.load(os.path.join(self.mnist_npy_dir, self.mode + '_images.npy'))
        labels = np.load(os.path.join(self.mnist_npy_dir, self.mode + '_labels.npy')).squeeze()
        self.input_data, self.input_label = images, labels
        if self.mode == 'train':
            self.shuffle_data()

    def get_data(self, batch_index):
        data = self.input_data[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        label = self.input_label[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        if self.mode == 'train':
            if len(data) == len(label) == self.batch_size:
                pass
            else:
                diff = self.batch_size - len(data)
                for _ in range(diff):
                    index = random.randint(1, len(self.input_data) - 1)
                    pad_data = self.input_data[index:index + 1]
                    pad_label = self.input_label[index:index + 1]
                    data += pad_data
                    label += pad_label
        data = np.array(data)
        label = np.array(label)

        return data, label
