#!/usr/bin/env python
# coding=utf-8

'Define the DatasetTool for social network experiments'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .basic_dataset_tool import BasicDatasetTool
from .helper_functions import safe_shuffle, vstack


class BatchData():
    def __init__(self, x, y):
        self.x = x.copy()
        self.y = y.copy()

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


class SocialDatasetTool(BasicDatasetTool):
    def __init__(self, root_folder='./data', payload='04'):
        super(SocialDatasetTool, self).__init__(root_folder=root_folder,
                                                var_type='test')
        self.payload = payload
        self.normal_images = None
        self.juniward_images = None
        self.nsF5_images = None
        self.UERD_images = None

    def load_images(self):
        # cover_80_resize1024
        self.normal_images = self.load_folder('cover_80')
        payload = self.payload
        # juniward_80_resize1024_04
        self.juniward_images = self.load_folder(
            'juniward_80_{}'.format(payload))
        # nsF5_80_resize1024_04
        self.nsF5_images = self.load_folder('nsF5_80_{}'.format(payload))
        # UERD_80_resize1024_04
        self.UERD_images = self.load_folder('UERD_80_{}'.format(payload))

    def load_folder(self, images_file_folder):
        result = None
        for i in range(1, 101):
            image_file_name = 'u_{i}'.format(i=i)
            image_file_path = '{data_path}/test/{images_file_folder}/{image_file_name}.mat'.format(
                data_path=self.root_folder,
                images_file_folder=images_file_folder,
                image_file_name=image_file_name)
            result = vstack(result,
                            sio.loadmat(image_file_path)[image_file_name])
        return result

    def dataset_target_1(self):
        result = []
        juniward_images = self.juniward_images.reshape(-1, 200, 320)
        for i in tqdm(range(100)):
            normal_dataset = safe_shuffle(
                self.normal_images.reshape(-1, 200, 320))
            normal_dataset[0] = juniward_images[i]

            batch_target = np.ones([100, 2])
            batch_target[:, 1] = -1
            batch_target[:1, :] = [-1, 1]

            result.append(BatchData(normal_dataset, batch_target))
        x = None
        y = None
        for batch_data in result:
            x = vstack(x, batch_data.get_x())
            y = vstack(y, batch_data.get_y())
        return x, y

    def dataset_target_2(self):
        result = []
        juniward_images = self.juniward_images.reshape(-1, 200, 320)
        nsF5_images = self.nsF5_images.reshape(-1, 200, 320)
        for i in tqdm(range(100)):
            normal_dataset = safe_shuffle(
                self.normal_images.reshape(-1, 200, 320))
            normal_dataset[0] = juniward_images[i]
            normal_dataset[1] = nsF5_images[i]

            batch_target = np.ones([100, 2])
            batch_target[:, 1] = -1
            batch_target[:2, :] = [-1, 1]

            result.append(BatchData(normal_dataset, batch_target))
        x = None
        y = None
        for batch_data in result:
            x = vstack(x, batch_data.get_x())
            y = vstack(y, batch_data.get_y())
        return x, y

    def dataset_target_3(self):
        result = []
        juniward_images = self.juniward_images.reshape(-1, 200, 320)
        nsF5_images = self.nsF5_images.reshape(-1, 200, 320)
        UERD_images = self.UERD_images.reshape(-1, 200, 320)
        for i in tqdm(range(100)):
            normal_dataset = safe_shuffle(
                self.normal_images.reshape(-1, 200, 320))
            normal_dataset[0] = juniward_images[i]
            normal_dataset[1] = nsF5_images[i]
            normal_dataset[2] = UERD_images[i]

            batch_target = np.ones([100, 2])
            batch_target[:, 1] = -1
            batch_target[:3, :] = [-1, 1]

            result.append(BatchData(normal_dataset, batch_target))
        x = None
        y = None
        for batch_data in result:
            x = vstack(x, batch_data.get_x())
            y = vstack(y, batch_data.get_y())
        return x, y

    def dataset(self, target_num):
        if target_num == 1:
            return self.dataset_target_1()
        elif target_num == 2:
            return self.dataset_target_2()
        elif target_num == 3:
            return self.dataset_target_3()
        else:
            raise RuntimeError('Target number can only be 1, 2 or 3')

    def save(self, target_folder='social_1', target_num=1):
        print('Generate Test Dataset:')
        self.load_images()
        print('Test Part:')
        self.x, self.y = self.dataset(target_num)
        self.check_save(target_folder)
