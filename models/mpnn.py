#!/usr/bin/env python
# coding: utf-8
# pylint: disable=C0103, C0111, E1101, W0612
"""Implementation of MPNN model."""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv, Set2Set

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
    def __init__(self,
                 positional_embedding_size=32,
                 max_node_freq=8,
                 max_edge_freq=8,
                 freq_embedding_size=32,
                 output_dim=32,
                 node_hidden_dim=32,
                 edge_hidden_dim=32,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(UnsupervisedMPNN, self).__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(positional_embedding_size + freq_embedding_size + 2, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(freq_embedding_size + 1, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum')
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

        self.max_node_freq = max_node_freq
        self.max_edge_freq = max_edge_freq

        self.node_freq_embedding = nn.Embedding(
                num_embeddings=max_node_freq+1,
                embedding_dim=freq_embedding_size)
        self.edge_freq_embedding = nn.Embedding(
                num_embeddings=max_edge_freq+1,
                embedding_dim=freq_embedding_size)

    def forward(self, g):
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

        nfreq = g.ndata['nfreq']
        n_feat = torch.cat(
                (
                    g.ndata['pos_undirected'],
                    g.ndata['pos_directed'],
                    self.node_freq_embedding(nfreq.clamp(0, self.max_node_freq)),
                    g.ndata['seed'].unsqueeze(1).float(),
                    nfreq.unsqueeze(1).float() / self.max_node_freq
                ),
                dim=-1
                )

        efreq = g.edata['efreq']
        e_feat = torch.cat(
                (
                    self.edge_freq_embedding(efreq.clamp(0, self.max_edge_freq)),
                    efreq.unsqueeze(1).float() / self.max_edge_freq
                ),
                dim=-1
                )

        out = F.relu(self.lin0(n_feat))                 # (B1, H1)
        h = out.unsqueeze(0)                            # (1, B1, H1)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(g, out, e_feat))       # (B1, H1)
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(g, out)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out

if __name__ == "__main__":
    model = UnsupervisedMPNN()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    g.ndata['pos_directed'] = torch.rand(3, 16)
    g.ndata['pos_undirected'] = torch.rand(3, 16)
    g.ndata['seed'] = torch.zeros(3, dtype=torch.long)
    g.ndata['nfreq'] = torch.ones(3, dtype=torch.long)
    g.edata['efreq'] = torch.ones(3, dtype=torch.long)
    y = model(g)
    print(y.shape)
    print(y)
