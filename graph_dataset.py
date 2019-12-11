#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import dgl
import torch
from dgl.data import AmazonCoBuy, Coauthor

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
    def __init__(self, rw_hops=128):
        self.rw_hops = rw_hops
        graphs = []
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            graphs.append(g)

        self.graph = dgl.batch(graphs, node_attrs=None, edge_attrs=None)

    def __len__(self):
        return self.graph.number_of_nodes()

    def __getitem__(self, idx):
        traces = dgl.contrib.sampling.random_walk(
                self.graph,
                seeds=[idx],
                num_traces=2,
                num_hops=self.rw_hops).view(2, -1)
        graph_q, graph_k = self.graph.subgraphs(traces)
        return graph_q, graph_k


if __name__ == '__main__':
	graph_dataset = GraphDataset()
	graph_loader = torch.utils.data.DataLoader(dataset=graph_dataset,
							batch_size=20,
							collate_fn=batcher(),
							shuffle=False,
							num_workers=4)

	for step, batch in enumerate(graph_loader):
		print(batch.graph_q.batch_size)
