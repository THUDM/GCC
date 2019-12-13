#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 15:38
# TODO:

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.model_zoo.chem.gnn import GCNLayer


class UnsupervisedGCN(nn.Module):
    def __init__(self, hidden_size=64, num_layer=2):
        super(UnsupervisedGCN, self).__init__()
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    hidden_size, hidden_size, F.relu if i + 1 < num_layer else nn.Sequential()
                )
                for i in range(num_layer)
            ]
        )

    def forward(self, g, feature):
        for layer in self.layers:
            feats = layer(g, feature)
        return feats


if __name__ == "__main__":
    model = UnsupervisedGCN()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    print(model(g, feat).shape)
