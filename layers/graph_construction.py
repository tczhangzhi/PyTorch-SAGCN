#!/usr/bin/env python
# coding=utf-8

'Define common layer'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanAdjConstruction(nn.Module):
    def __init__(self):
        super(EuclideanAdjConstruction, self).__init__()

    def forward(self, x):
        dim = x.dim()
        distance = torch.norm(x.unsqueeze(dim - 2) - x.unsqueeze(dim - 1),
                              dim=dim)
        return torch.exp(-distance).sqrt()


class CosineAdjConstruction(nn.Module):
    def __init__(self):
        super(CosineAdjConstruction, self).__init__()

    def forward(self, x):
        target_a = x.unsqueeze(-3)
        target_a_shape = list(target_a.shape)
        target_a_shape[-3] = x.size(-2)
        target_a = target_a.expand(target_a_shape)

        target_b = x.unsqueeze(-2)
        target_b_shape = list(target_a.shape)
        target_b_shape[-2] = x.size(-2)
        target_b = target_b.expand(target_b_shape)

        distance = F.cosine_similarity(target_a, target_b, dim=x.dim())
        # pdb.set_trace()
        return torch.exp(-distance).sqrt()


def get_adj_construction_func(name):
    if name == 'Euclidean':
        return EuclideanAdjConstruction
    elif name == 'Cosine':
        return CosineAdjConstruction
    else:
        raise RuntimeError('Embedding function does not exist!')