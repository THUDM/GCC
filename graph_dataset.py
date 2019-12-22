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
import dgl.data
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from itertools import accumulate
import sklearn.preprocessing as preprocessing

class GraphBatcher:
    def __init__(self, graph_q, graph_k, graph_q_roots, graph_k_roots):
        self.graph_q = graph_q
        self.graph_k = graph_k
        self.graph_q_roots = graph_q_roots
        self.graph_k_roots = graph_k_roots

def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q_roots = list(accumulate([0] + [len(graph) for graph in graph_q[:-1]]))
        graph_k_roots = list(accumulate([0] + [len(graph) for graph in graph_k[:-1]]))
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return GraphBatcher(graph_q, graph_k, graph_q_roots, graph_k_roots)
    return batcher_dev

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, rw_hops=2048, subgraph_size=128, restart_prob=0.6, hidden_size=64):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.hidden_size = hidden_size
        assert(hidden_size > 1)
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/graphs.bin")
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

        #  self.graph = dgl.batch(graphs, node_attrs=None, edge_attrs=None)
        #  self.graph.readonly()
        #  print(self.graph.number_of_nodes(), self.graph.number_of_edges())

    def add_graph_features(self, g, retry=10):
        # We use sigular vectors of normalized graph laplacian as vertex features.
        # It could be viewed as a generalization of positional embedding in the
        # attention is all you need paper.
        # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
        # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
        n = g.number_of_nodes()
        adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
        norm = sparse.diags(
                dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5,
                dtype=float)
        laplacian = norm * adj * norm

        k=min(n-1, self.hidden_size-1)
        for i in range(retry):
            try:
                s, u = eigsh(
                        laplacian,
                        k=k,
                        which='LA',
                        ncv=n)
            except sparse.linalg.eigen.arpack.ArpackError:
                print("arpack error, retry=", i)
                if i + 1 == retry:
                    sparse.save_npz('arpack_error_sparse_matrix.npz', laplacian)
                    exit()
                    x = torch.zeros(g.number_of_nodes(), self.hidden_size)
            else:
                x = preprocessing.normalize(u, norm='l2')
                break
        #  x = u * sparse.diags(np.sqrt(np.abs(s)))
        x = torch.from_numpy(x)
        x = F.pad(x, (0, self.hidden_size-k), 'constant', 0)
        g.ndata['x'] = x.float()
        g.ndata['x'][0, -1] = 1.0

        # TODO netmf can also be part of vertex features
        return g

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = dgl.contrib.sampling.deepinf_random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            num_traces=2,
            num_hops=self.rw_hops,
            num_unique=self.subgraph_size)[0]
        assert traces[0][0].item() == node_idx, traces[1][0].item() == node_idx

        graph_q, graph_k = self.graphs[graph_idx].subgraphs(traces) # equivalent to x_q and x_k in moco paper
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
        print(batch.graph_q)
        print(batch.graph_q.ndata['x'].shape)
        print(batch.graph_q.batch_size)
        break
