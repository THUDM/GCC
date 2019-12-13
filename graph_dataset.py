#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import dgl
import torch
import torch.nn.functional as F
from dgl.data import AmazonCoBuy, Coauthor
import scipy.sparse as sparse
import numpy as np

class GraphBatcher:
    def __init__(self, graph_q, graph_k):
        self.graph_q = graph_q
        self.graph_k = graph_k

def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return GraphBatcher(graph_q, graph_k)
    return batcher_dev

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, rw_hops=2048, subgraph_size=128, restart_prob=0.6, hidden_size=64):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.hidden_size = hidden_size
        graphs = []
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            graphs.append(g)
        # more graphs are comming ...

        self.graph = dgl.batch(graphs, node_attrs=None, edge_attrs=None)
        self.graph.readonly()

    def add_graph_features(self, g):
        # We use sigular vectors of normalized graph laplacian as vertex features.
        # It could be viewed as a generalization of positional embedding in the
        # attention is all you need paper.
        # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
        # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
        n = g.number_of_nodes()
        adj = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        norm = sparse.diags(
                dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5,
                dtype=float)
        laplacian = norm * adj * norm

        u, s, _ = sparse.linalg.svds(
                laplacian,
                k=min(n-1, self.hidden_size),
                which='LM',
                return_singular_vectors='u')
        x = u * sparse.diags(np.sqrt(s))
        x = torch.from_numpy(x)
        if n - 1 < self.hidden_size:
            x = F.pad(x, (0, self.hidden_size-n+1), 'constant', 0)
        g.ndata['x'] = x

        # TODO netmf can also be part of vertex features
        return g

    def __len__(self):
        return self.graph.number_of_nodes()

    def __getitem__(self, idx):
        traces = dgl.contrib.sampling.deepinf_random_walk_with_restart(
            self.graph,
            seeds=[idx],
            restart_prob=self.restart_prob,
            num_traces=2,
            num_hops=self.rw_hops,
            num_unique=self.subgraph_size)[0]

        graph_q, graph_k = self.graph.subgraphs(traces) # equivalent to x_q and x_k in moco paper
        graph_q = self.add_graph_features(graph_q)
        graph_k = self.add_graph_features(graph_k)
        return graph_q, graph_k


if __name__ == '__main__':
    graph_dataset = GraphDataset()
    graph_loader = torch.utils.data.DataLoader(
            dataset=graph_dataset,
            batch_size=20,
            collate_fn=batcher(),
            shuffle=False,
            num_workers=4)
    for step, batch in enumerate(graph_loader):
        print(batch.graph_q.batch_size)
        break
