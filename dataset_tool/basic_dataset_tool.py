#!/usr/bin/env python
# coding=utf-8

'Define a basic class of DatasetTool with common functions'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import pickle as pkl

from os import path, makedirs
from shutil import rmtree


class BasicDatasetTool:
    def __init__(self, root_folder, var_type):
        self.root_folder = root_folder
        self.var_type = var_type
        self.x = None
        self.y = None

    def check_target_folder(self, target_folder):
        key_words = ['train', 'test']
        if target_folder in key_words:
            raise RuntimeError('Proscribed path of folder_name!')

        target_path = '{data_path}/{target}'.format(data_path=self.root_folder,
                                                    target=target_folder)
        if not path.exists(target_path):
            makedirs(target_path)

    @staticmethod
    def reset_target_folder(target_folder, root_folder='./data'):
        target_path = '{data_path}/{target}'.format(data_path=root_folder,
                                                    target=target_folder)
        if path.exists(target_path):
            rmtree(target_path)
        makedirs(target_path)

    def save_target_files(self, target_folder, alias_type):
        if not alias_type:
            alias_type = self.var_type
        with open(
                '{data_path}/{target_folder}/{var_type}_x'.format(
                    data_path=self.root_folder,
                    target_folder=target_folder,
                    var_type=alias_type), 'wb+') as f:
            pkl.dump(self.x, f, protocol=4)
        with open(
                '{data_path}/{target_folder}/{var_type}_y'.format(
                    data_path=self.root_folder,
                    target_folder=target_folder,
                    var_type=alias_type), 'wb+') as f:
            pkl.dump(self.y, f, protocol=4)

    def check_save(self, target_folder, alias_type=''):
        self.check_target_folder(target_folder)
        self.save_target_files(target_folder, alias_type)
        self.x = None
        self.y = None