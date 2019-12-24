#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import torch
import torch.nn.functional as F
from dgl.data import AmazonCoBuy, Coauthor
import dgl.data
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from itertools import accumulate
import sklearn.preprocessing as preprocessing

from cogdl.datasets import build_dataset


def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call."""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	plt.close(figure)
	buf.seek(0)
	# Convert PNG buffer to TF image
	image = tf.image.decode_png(buf.getvalue(), channels=3)
	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	return torch.from_numpy(image.numpy())

def _rwr_trace_to_dgl_graph(g, seed, trace, hidden_size):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    subg = g.subgraph(subv)
    assert subg.parent_nid[0] == seed, "by construction, node 0 in subgraph should be the seed"

    subg = _add_graph_features(subg, hidden_size)

    mapping = dict([(v, k) for k, v in enumerate(subg.parent_nid.tolist())])
    visit_count = torch.zeros(subg.number_of_edges(), dtype=torch.long)
    for walk in trace:
        u = seed
        for v in walk.tolist():
            # add edge feature for (u, v)
            eid = subg.edge_id(mapping[u], mapping[v])
            visit_count[eid] += 1
            u = v
    subg.edata['efeat'] = visit_count
    return subg

def _add_graph_features(g, hidden_size, retry=10):
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

    k=min(n-1, hidden_size-1)
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
                x = torch.zeros(g.number_of_nodes(), hidden_size)
        else:
            x = preprocessing.normalize(u, norm='l2')
            break
    #  x = u * sparse.diags(np.sqrt(np.abs(s)))
    x = torch.from_numpy(x)
    x = F.pad(x, (0, hidden_size-k), 'constant', 0)
    g.ndata['x'] = x.float()
    g.ndata['x'][0, -1] = 1.0

    # TODO netmf can also be part of vertex features
    return g

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
    def __init__(self, rw_hops=64, subgraph_size=64, restart_prob=0.8, hidden_size=32, step_dist=[1.0, 0.0, 0.0]):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.hidden_size = hidden_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert(hidden_size > 1)
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs("data_bin/dgl/lscc_graphs.bin", [0, 1, 2])
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


    def __len__(self):
        return self.length

    def getplot(self, idx):
        graph_q, graph_k = self.__getitem__(idx)
        graph_q = graph_q.to_networkx()
        graph_k = graph_k.to_networkx()
        figure_q = plt.figure(figsize=(10, 10))
        nx.draw(graph_q)
        plt.draw()
        image_q = plot_to_image(figure_q)
        figure_k = plt.figure(figsize=(10, 10))
        nx.draw(graph_k)
        plt.draw()
        image_k = plot_to_image(figure_k)
        return image_q, image_k


    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if  node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                    g=self.graphs[graph_idx],
                    seeds=[node_idx],
                    num_traces=1,
                    num_hops=step
                    )[0][0][-1].item()

        #  traces = dgl.contrib.sampling.deepinf_random_walk_with_restart(
        #      self.graphs[graph_idx],
        #      seeds=[node_idx],
        #      restart_prob=self.restart_prob,
        #      num_traces=2,
        #      num_hops=self.rw_hops,
        #      num_unique=self.subgraph_size)[0]
        #  assert traces[0][0].item() == node_idx, traces[1][0].item() == node_idx
        #  graph_q, graph_k = self.graphs[graph_idx].subgraphs(traces) # equivalent to x_q and x_k in moco paper
        #  graph_q = _add_graph_features(graph_q, hidden_size=self.hidden_size)
        #  graph_k = _add_graph_features(graph_k, hidden_size=self.hidden_size)

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops)

        graph_q = _rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                hidden_size=self.hidden_size)
        graph_k = _rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=other_node_idx,
                trace=traces[1],
                hidden_size=self.hidden_size)
        return graph_q, graph_k


class CogDLGraphDataset(GraphDataset):
    def __init__(self, dataset, rw_hops=64, subgraph_size=64, restart_prob=0.8, hidden_size=32, step_dist=[1.0, 0.0, 0.0]):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.hidden_size = hidden_size
        self.step_dist = step_dist
        assert(hidden_size > 1)

        class tmp():
            # HACK
            pass
        args = tmp()
        args.dataset = dataset
        data = build_dataset(args)[0]
        self.graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        self.graph.add_nodes(data.y.shape[0])
        self.graph.add_edges(src, dst)
        # self.graph = dgl.transform.remove_self_loop(self.graph)
        self.graph.remove_nodes((self.graph.out_degrees() == 0).nonzero().squeeze())
        self.graph.readonly()
        self.graphs = [self.graph]
        self.length = sum([g.number_of_nodes() for g in self.graphs])

if __name__ == '__main__':
    # graph_dataset = GraphDataset()
    graph_dataset = CogDLGraphDataset(dataset="wikipedia")
    pq, pk = graph_dataset.getplot(0)
    graph_loader = torch.utils.data.DataLoader(
            dataset=graph_dataset,
            batch_size=20,
            collate_fn=batcher(),
            shuffle=True,
            num_workers=4)
    for step, batch in enumerate(graph_loader):
        print(batch.graph_q)
        print(batch.graph_q.ndata['x'].shape)
        print(batch.graph_q.batch_size)
        print("max", batch.graph_q.edata['efeat'].max())
        break
