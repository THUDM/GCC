#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:


import torch
import tensorflow as tf
import io
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import sklearn.preprocessing as preprocessing
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt
from itertools import accumulate

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
    # We use eigenvectors of normalized graph laplacian as vertex features.
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

