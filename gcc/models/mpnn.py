#!/usr/bin/env python
# coding: utf-8
# pylint: disable=C0103, C0111, E1101, W0612
"""Implementation of MPNN model."""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv


class UnsupervisedMPNN(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        output_dim=32,
        node_input_dim=32,
        node_hidden_dim=32,
        edge_input_dim=32,
        edge_hidden_dim=32,
        num_step_message_passing=6,
        lstm_as_gate=False,
    ):
        super(UnsupervisedMPNN, self).__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim),
        )
        self.conv = NNConv(
            in_feats=node_hidden_dim,
            out_feats=node_hidden_dim,
            edge_func=edge_network,
            aggregator_type="sum",
        )
        self.lstm_as_gate = lstm_as_gate
        if lstm_as_gate:
            self.lstm = nn.LSTM(node_hidden_dim, node_hidden_dim)
        else:
            self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

    def forward(self, g, n_feat, e_feat):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """
        out = F.relu(self.lin0(n_feat))  # (B1, H1)
        h = out.unsqueeze(0)  # (1, B1, H1)
        c = torch.zeros_like(h)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(g, out, e_feat))  # (B1, H1)
            if self.lstm_as_gate:
                out, (h, c) = self.lstm(m.unsqueeze(0), (h, c))
            else:
                out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        return out


if __name__ == "__main__":
    model = UnsupervisedMPNN()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    g.ndata["pos_directed"] = torch.rand(3, 16)
    g.ndata["pos_undirected"] = torch.rand(3, 16)
    g.ndata["seed"] = torch.zeros(3, dtype=torch.long)
    g.ndata["nfreq"] = torch.ones(3, dtype=torch.long)
    g.edata["efreq"] = torch.ones(3, dtype=torch.long)
    y = model(g)
    print(y.shape)
    print(y)
