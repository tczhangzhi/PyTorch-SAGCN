#!/usr/bin/env python
# coding=utf-8

'Define model'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import torch.nn as nn
import torch.nn.functional as F
from layers.common import Flatten
from layers.graph_construction import get_adj_construction_func
from layers.graph_convolution import GraphConv1d


class SAGCN(nn.Module):
    def __init__(self, embedding_func='Euclidean'):
        super(SAGCN, self).__init__()

        self.downs = nn.Sequential(nn.Linear(320, 160), nn.ReLU(),
                                   nn.Linear(160, 80), nn.ReLU())

        GEmbedding = get_adj_construction_func(embedding_func)

        self.convs = nn.Sequential(
            GraphConv1d(80, 20, GEmbedding, 0.05, 0.05, 0.05),
            GraphConv1d(20, 5, GEmbedding, 0.1, 0.1, 0.1))

        self.fc = nn.Sequential(Flatten(), nn.Linear(1000, 2))

    def forward(self, x):
        x = self.downs(x)
        x = self.convs(x)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
