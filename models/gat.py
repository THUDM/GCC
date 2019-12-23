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
from dgl.model_zoo.chem.gnn import GATLayer
from dgl.nn.pytorch import AvgPooling, Set2Set


class UnsupervisedGAT(nn.Module):
    def __init__(self, hidden_size=64, num_layer=2, readout='avg', num_heads=4,
            layernorm: bool = False,
            set2set_lstm_layer: int = 3, set2set_iter: int = 6
            ):
        super(UnsupervisedGAT, self).__init__()
        self.layers = nn.ModuleList(
            [
                GATLayer(
                    in_feats=hidden_size,
                    out_feats=hidden_size // 4,
                    num_heads=4,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    alpha=0.2,
                    residual=False,
                    agg_mode='flatten',
                    activation=F.leaky_relu if i + 1 < num_layer else None
                )
                for i in range(num_layer)
            ]
        )
        if readout == "avg":
            self.readout = AvgPooling()
        elif readout == "set2set":
            self.readout = Set2Set(hidden_size, n_iters=set2set_iter, n_layers=set2set_lstm_layer)
            self.linear = nn.Linear(2 * hidden_size, hidden_size)
        elif readout == "root":
            # HACK: process outside the model part
            self.readout = lambda _, x: x
        else:
            raise NotImplementedError

        self.layernorm = layernorm
        if layernorm:
            self.lns = nn.ModuleList(
                [nn.LayerNorm(hidden_size, elementwise_affine=True)
                    for i in range(num_layer + 1)])

    def forward(self, g, feats):
        for i, layer in enumerate(self.layers):
            feats = layer(g, feats)
            if self.layernorm:
                feats = self.lns[i](feats)
        feats = self.readout(g, feats)
        if isinstance(self.readout, Set2Set):
            feats = self.linear(feats)
        if self.layernorm:
            feats = self.lns[-1](feats)
        return feats


if __name__ == "__main__":
    model = UnsupervisedGAT()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    print(model(g, feat).shape)
