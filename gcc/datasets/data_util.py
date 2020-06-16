#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:

import io
import itertools
import os
import os.path as osp
from collections import defaultdict, namedtuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg


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


Data = namedtuple("Data", ["x", "edge_index", "y"])


def create_graph_classification_dataset(dataset_name):
    name = {
        "imdb-binary": "IMDB-BINARY",
        "imdb-multi": "IMDB-MULTI",
        "rdt-b": "REDDIT-BINARY",
        "rdt-5k": "REDDIT-MULTI-5K",
        "collab": "COLLAB",
    }[dataset_name]
    dataset = TUDataset(name)
    dataset.num_labels = dataset.num_labels[0]
    dataset.graph_labels = dataset.graph_labels.squeeze()
    return dataset


class Edgelist(object):
    def __init__(self, root, name):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


def create_node_classification_dataset(dataset_name):
    if "airport" in dataset_name:
        return Edgelist(
            "data/struc2vec/",
            {
                "usa_airport": "usa-airports",
                "brazil_airport": "brazil-airports",
                "europe_airport": "europe-airports",
            }[dataset_name],
        )
    elif "h-index" in dataset_name:
        return Edgelist(
            "data/hindex/",
            {
                "h-index-rand-1": "aminer_hindex_rand1_5000",
                "h-index-top-1": "aminer_hindex_top1_5000",
                "h-index": "aminer_hindex_rand20intop200_5000",
            }[dataset_name],
        )
    else:
        raise NotImplementedError


def _rwr_trace_to_dgl_graph(
    g, seed, trace, positional_embedding_size, entire_graph=False
):
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

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g
