#!/usr/bin/env python
# coding=utf-8

'Dataset utils'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import pickle as pkl
import sys

from torch.utils.data import Dataset


class SteganographyDataset(Dataset):
    """Load dataset and prepare for pytorch training"""
    def __init__(self,
                 root_folder='./data',
                 folder_name='example',
                 dataset='train'):
        self.root_folder = root_folder
        datasets = ['train', 'val', 'test']
        if dataset not in datasets:
            raise RuntimeError('Dataset not found!')
        variables = ['x', 'y']
        data = []
        for i in range(len(variables)):
            with open(
                    "{data_path}/{folder_name}/{dataset}_{name}".format(
                        data_path=self.root_folder,
                        folder_name=folder_name,
                        dataset=dataset,
                        name=variables[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    data.append(pkl.load(f, encoding='latin1'))
                else:
                    data.append(pkl.load(f))
        x, y = tuple(data)

        if len(x) != len(y):
            raise RuntimeError('Wrong feature or label size!')

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
