#!/usr/bin/env python
# coding=utf-8

'Define layer'

__author__ = 'Zhi Zhang, Mingjie Zheng, Sheng hua Zhong, Yan Liu'

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class GraphAdjNorm(nn.Module):
    def __init__(self):
        super(GraphAdjNorm, self).__init__()

    def forward(self, x):
        eye_helper = torch.eye(x.size(-2), x.size(-1), device=x.device)
        # D
        d = torch.sum(x, dim=-1, keepdim=True)
        # D^-0.5
        inv_sqrt = d.sqrt().reciprocal() * eye_helper
        # D^-0.5A
        result = torch.matmul(inv_sqrt, x)
        # D^-0.5AD^-0.5
        result = torch.matmul(result, inv_sqrt)
        return result


class GraphConv1d(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 GEmbedding,
                 t1=0.05,
                 t2=0.10,
                 t3=0.15):
        super(GraphConv1d, self).__init__()

        self.embd = GEmbedding()
        self.norm = GraphAdjNorm()

        self.in_features = in_features
        self.out_features = out_features

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.weight.data)

    def forward(self, x):
        res = x.clone()

        with torch.no_grad():
            adj = self.embd(x)
            adj = F.threshold(adj, self.t1, 0)
            adj = self.norm(adj)
            x = torch.matmul(adj, x)

            adj = self.embd(x)
            adj = F.threshold(adj, self.t2, 0)
            adj = self.norm(adj)
            x = torch.matmul(adj, x)

            adj = self.embd(x)
            adj = F.threshold(adj, self.t3, 0)
            adj = self.norm(adj)
            x = torch.matmul(adj, x)

        x = x + res
        x = torch.matmul(x, self.weight)
        return x