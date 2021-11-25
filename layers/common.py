#!/usr/bin/env python
# coding=utf-8

'Define layer'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import pdb
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mean(nn.Module):
    def forward(self, x):
        return x.mean(dim=-2)


class Debugger(nn.Module):
    def forward(self, x):
        pdb.set_trace()
        return x