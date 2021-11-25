#!/usr/bin/env python
# coding=utf-8

'Define the DatasetTool for spatial/frequency domain experiments in the test stage'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .basic_dataset_tool import BasicDatasetTool
from .helper_functions import mix_normal, safe_batch_data, vstack


class TestDatasetTool(BasicDatasetTool):
    def __init__(self, root_folder='./data'):
        super(TestDatasetTool, self).__init__(root_folder=root_folder,
                                              var_type='test')
        self.normal_images = None
        self.guilty_images = None

    def load_images(self, normal_images_file, guilty_images_file):
        self.normal_images = sio.loadmat(
            '{data_path}/test/{normal_images_file}.mat'.format(
                data_path=self.root_folder,
                normal_images_file=normal_images_file))[normal_images_file]
        self.guilty_images = sio.loadmat(
            '{data_path}/test/{guilty_images_file}.mat'.format(
                data_path=self.root_folder,
                guilty_images_file=guilty_images_file))[guilty_images_file]

    def dataset_x(self, batch, mixin_num):
        result = None
        guilty_dataset = safe_batch_data(self.guilty_images)
        if mixin_num:
            mix_normal_dataset = safe_batch_data(self.normal_images)
            guilty_dataset = mix_normal(guilty_dataset, mix_normal_dataset,
                                        mixin_num)
        for i in tqdm(range(batch)):
            normal_dataset = safe_batch_data(self.normal_images)
            normal_dataset[0] = guilty_dataset[i]
            result = vstack(result, normal_dataset)
        return result

    def dataset_y(self, batch):
        result = None
        for _ in tqdm(range(batch)):
            batch_result = np.ones([100, 2])
            batch_result[:, 1] = -1
            batch_result[0, :] = [-1, 1]
            result = vstack(result, batch_result)
        return result

    def save(self,
             normal_images_file='cover',
             guilty_images_file='suniward_04',
             batch=100,
             mixin_num=0,
             target_folder='suniward_04_100'):
        print('Generate Test Dataset:')
        self.load_images(normal_images_file, guilty_images_file)
        print('Train Dataset Test X Part:')
        self.x = self.dataset_x(batch, mixin_num)
        print('Train Dataset Test Y Part:')
        self.y = self.dataset_y(batch)
        self.check_save(target_folder)
