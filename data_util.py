#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:

import numpy as np
import scipy
import torch
# import tensorflow as tf
import io
import scipy.sparse as sparse
from scipy.sparse import linalg
import sklearn.preprocessing as preprocessing
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt
import itertools

class Distance(object):
    def __init__(self, p=2, emb_name="prone"):
        self.p = p # p-norm
        self.emb_name = emb_name
    def __call__(self, graph):
        def distance(edges):
            return {'dis': torch.norm(edges.src[self.emb_name]-edges.dst[self.emb_name], p=self.p, dim=1)}
        graph.apply_edges(func=distance,
                edges='__ALL__',
                inplace=True)
        return graph

def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k
    return batcher_dev

def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)
    return batcher_dev

class dynamic_batcher(object):
    def __init__(self, max_node_per_batch=128*32*2, max_edge_per_batch=20000):
        self.max_node_per_batch = max_node_per_batch
        self.max_edge_per_batch = max_edge_per_batch

    def __call__(self, batch):
        # TODO make it more elegant with itertools?
        #  batch = sorted(batch, key=lambda graphs: graphs[0].number_of_edges() + graphs[1].number_of_edges())
        graph_q, graph_k = zip(*batch)
        accum_node, accum_edge = 0, 0
        for i in range(len(graph_q)):
            accum_node += graph_q[i].number_of_nodes() + graph_k[i].number_of_nodes()
            accum_edge += graph_q[i].number_of_edges() + graph_k[i].number_of_edges()
            if i > 1 and (accum_node > self.max_node_per_batch or accum_edge > self.max_edge_per_batch):
                graph_q = graph_q[:i]
                graph_k = graph_k[:i]
                break
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k

class filter_batcher(object):
    def __init__(self, max_node=256, max_edge=2048):
        self.max_node = max_node
        self.max_edge = max_edge

    def __call__(self, batch):
        predicate = lambda graphs: \
                graphs[0].number_of_nodes() > self.max_node or \
                graphs[0].number_of_edges() > self.max_edge or \
                graphs[1].number_of_nodes() > self.max_node or \
                graphs[1].number_of_edges() > self.max_edge
        batch = itertools.filterfalse(predicate, batch)
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k

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

def _edge_subgraph(trace, seed):
    mapping = dict()
    edge_list = set()
    mapping[seed] = 0
    for walk in trace:
        u = seed
        for v in walk.tolist():
            if (u, v) not in edge_list:
                if u not in mapping:
                    mapping[u] = len(mapping)
                if v not in mapping:
                    mapping[v] = len(mapping)
                edge_list.add((u, v))
            u = v
    subg = dgl.DGLGraph()
    subg.add_nodes(len(mapping))
    u_list, v_list = [], []
    for u, v in edge_list:
        u_list.append(mapping[u])
        v_list.append(mapping[v])
    subg.add_edges(u_list, v_list)
    return subg

def _rwr_trace_to_dgl_graph(g, seed, trace, positional_embedding_size, entire_graph=False):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)
    # assert subg.parent_nid[0] == seed, "by construction, node 0 in subgraph should be the seed"

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)
    # subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size // 2)

    # mapping = dict([(v, k) for k, v in enumerate(subg.parent_nid.tolist())])
    # nfreq = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    # efreq = torch.zeros(subg.number_of_edges(), dtype=torch.long)

    # M = np.zeros(
    #         shape=(subg.number_of_nodes(), subg.number_of_nodes()),
    #         dtype=np.float32
    #         )
    # for walk in trace:
    #     u = mapping[seed]
    #     nfreq[u] += 1
    #     for v in walk.tolist():
    #         v = mapping[v]
    #         nfreq[v] += 1
    #         # add edge feature for (u, v)
    #         eid = subg.edge_id(u, v)
    #         efreq[eid] += 1
    #         M[u, v] += 1
    #         u = v

    # subg = _add_directed_graph_positional_embedding(subg, M, positional_embedding_size // 2)

    # subg.ndata['nfreq'] = nfreq
    # subg.edata['efreq'] = efreq

    subg.ndata['seed'] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata['seed'][seed] = 1
    else:
        subg.ndata['seed'][0] = 1
    return subg

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    ncv=min(n, max(2*k + 1, 20))
    for i in range(retry):
        try:
            s, u = linalg.eigsh(
                    laplacian,
                    k=k,
                    which='LA',
                    ncv=ncv)
        except sparse.linalg.eigen.arpack.ArpackError:
            print("arpack error, retry=", i)
            ncv = min(ncv*2, n)
            if i + 1 == retry:
                sparse.save_npz('arpack_error_sparse_matrix.npz', laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm='l2')
    x = torch.from_numpy(x)
    x = F.pad(x, (0, hidden_size-k), 'constant', 0)
    return x


def _add_directed_graph_positional_embedding(g, M, hidden_size, retry=10, alpha=0.95):
    # Follow https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.linalg.laplacianmatrix.directed_laplacian_matrix.html#directed-laplacian-matrix
    # We use its pagerank mode
    n = g.number_of_nodes()
    # add constant to dangling nodes' row
    dangling = scipy.where(M.sum(axis=1) == 0)
    for d in dangling[0]:
        M[d] = 1.0 / n
    # normalize
    M = M / M.sum(axis=1)
    P = alpha * M + (1 - alpha) / n
    if n == 2:
        evals, evecs = np.linalg.eig(P.T)
        evals = evals.flatten().real
        evecs = evecs[:, 0] if evals[0] > evals[1] else evecs[:, 1]
    else:
        evals, evecs = sparse.linalg.eigs(P.T, k=1, ncv=n)
    v = evecs.flatten().real
    p =  v / v.sum()
    sqrtp = scipy.sqrt(p)
    Q = sparse.spdiags(sqrtp, [0], n, n) * P * sparse.spdiags(1.0/sqrtp, [0], n, n)
    #  I = scipy.identity(n)
    #  laplacian = I - (Q + Q.T)/2.0
    laplacian = (Q + Q.T)/2.0
    k=min(n-2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata['pos_directed'] = x.float()
    return g

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
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
    k=min(n-2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata['pos_undirected'] = x.float()
    return g

