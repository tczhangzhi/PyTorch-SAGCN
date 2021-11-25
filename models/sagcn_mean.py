#!/usr/bin/env python
# coding=utf-8

'Define model'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import torch.nn as nn
import torch.nn.functional as F
from layers.common import Mean
from layers.graph_construction import get_adj_construction_func
from layers.graph_convolution import GraphConv1d


class SAGCNMean(nn.Module):
    def __init__(self, embedding_func='Euclidean'):
        super(SAGCNMean, self).__init__()

        self.downs = nn.Sequential(nn.Linear(320, 160), nn.ReLU(),
                                   nn.Linear(160, 80), nn.ReLU())

        GEmbedding = get_adj_construction_func(embedding_func)

        self.convs = nn.Sequential(GraphConv1d(80, 40, GEmbedding),
                                   GraphConv1d(40, 5, GEmbedding))

        self.fc = nn.Sequential(Mean(), nn.Linear(5, 2))

    def forward(self, x):
        x = self.downs(x)
        x = self.convs(x)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
