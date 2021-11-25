#!/usr/bin/env python
# coding=utf-8

'Define the DatasetTool for spatial/frequency domain experiments in the training stage'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .basic_dataset_tool import BasicDatasetTool
from .helper_functions import dataset, mix_normal, vstack


class TrainDatasetTool(BasicDatasetTool):
    def __init__(self, root_folder='./data'):
        super(TrainDatasetTool, self).__init__(root_folder=root_folder,
                                               var_type='train')
        self.normal_images = None
        self.guilty_images = None

    def load_images(self, normal_images_file, guilty_images_file):
        self.normal_images = sio.loadmat(
            '{data_path}/train/{normal_images_file}.mat'.format(
                data_path=self.root_folder,
                normal_images_file=normal_images_file))[normal_images_file]
        self.guilty_images = sio.loadmat(
            '{data_path}/train/{guilty_images_file}.mat'.format(
                data_path=self.root_folder,
                guilty_images_file=guilty_images_file))[guilty_images_file]

    def dataset_x(self, normal_batch, guilty_batch, mixin_num):
        result = None
        result = vstack(result, dataset(self.normal_images, normal_batch))
        if mixin_num:
            result = vstack(
                result,
                mix_normal(dataset(self.guilty_images, guilty_batch),
                           dataset(self.normal_images, guilty_batch),
                           mixin_num))
        else:
            result = vstack(result, dataset(self.guilty_images, guilty_batch))
        return result

    def dataset_y(self, normal_batch, guilty_batch):
        result = None
        for _ in tqdm(range(normal_batch)):
            batch_result = np.ones([100, 2])
            batch_result[:, 1] = -1
            result = vstack(result, batch_result)
        for _ in tqdm(range(guilty_batch)):
            batch_result = np.ones([100, 2])
            batch_result[:, 0] = -1
            result = vstack(result, batch_result)
        return result

    def save(self,
             normal_images_file='cover',
             guilty_images_file='suniward_04',
             normal_batch=50,
             guilty_batch=50,
             mixin_num=0,
             target_folder='suniward_04_100'):
        print('Generate Train Dataset:')
        self.load_images(normal_images_file, guilty_images_file)
        print('Train Dataset Train X Part:')
        self.x = self.dataset_x(normal_batch, guilty_batch, mixin_num)
        print('Train Dataset Train Y Part:')
        self.y = self.dataset_y(normal_batch, guilty_batch)
        self.check_save(target_folder)
        print('Train Dataset Val X Part:')
        self.x = self.dataset_x(normal_batch, guilty_batch, mixin_num)
        print('Train Dataset Val Y Part:')
        self.y = self.dataset_y(normal_batch, guilty_batch)
        self.check_save(target_folder, alias_type='val')
