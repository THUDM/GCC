#!/usr/bin/env python
# encoding: utf-8
# File Name: x2dgl.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/20 13:39
# TODO:

import re
import dgl
from dgl.data.utils import save_graphs
import numpy as np
import scipy.sparse as sp
import argparse
import pathlib
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def yuxiao_kdd17_graph_to_dgl(file):
    logger.info("processing %s", file)
    with open(file, "r") as f:
        n = int(f.readline().split()[1])
        for i in range(n):
            # this line include zero-bazed vertex index, and their index in the raw data
            # so we do not need them
            f.readline()
        m = int(f.readline().split()[1])
        m_true = 0
        row, col = [0] * (2*m), [0] * (2*m)
        for i in range(m):
            u, v, _ = list(map(int, f.readline().split()))
            if u == v:
                continue
            u, v = min(u, v), max(u, v)
            row[m_true], col[m_true] = u, v
            m_true += 1
            row[m_true], col[m_true] = v, u
            m_true += 1

    logger.info("raw file has %d nodes and %d edges", n, m)
    val = [1] * m_true
    row, col = row[:m_true], col[:m_true]
    A = sp.coo_matrix((val, (row, col)), shape=(n, n))
    A = A.tocsr()
    # tocsr will deduplicate edges, and sum up their value
    A.data = np.ones_like(A.data) # delete this line if not necessary
    sym_err = A - A.T
    sym_check_res = np.all(np.abs(sym_err.data) < 1e-10)  # tune this value
    assert sym_check_res, 'input matrix is not symmetric!!!'
    g = dgl.DGLGraph()
    g.from_scipy_sparse_matrix(A)
    g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
    logger.info("%d nodes, %d edges, %d self-loop(s) removed, %d zero-degree node(s) removed", g.number_of_nodes(), g.number_of_edges(), 2*m-m_true, n-g.number_of_nodes())
    logger.info("return graph %s", re.sub("\s+", " ", str(g)))
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for x2dgl")
    parser.add_argument("--graph-dir", type=str, default=None, help="dir to load graphs")
    parser.add_argument("--save-file", type=str, default=None, help="file to save graphs")

    args = parser.parse_args()
    graph_dir = pathlib.Path(args.graph_dir)
    #  g = yuxiao_kdd17_graph_to_dgl("data_bin/ca-GrQc-SNAP.txtu.lpm.lscc")
    graphs = [yuxiao_kdd17_graph_to_dgl(graph_file) for graph_file in graph_dir.iterdir() if graph_file.is_file()]
    logger.info("save graphs to %s", args.save_file)
    save_graphs(args.save_file, graphs)
