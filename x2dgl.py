#!/usr/bin/env python
# encoding: utf-8
# File Name: x2dgl.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/20 13:39
# TODO:

import dgl
from dgl.data.utils import save_graphs
import numpy as np
import scipy.sparse as sp
import argparse
import pathlib

def yuxiao_kdd17_graph_to_dgl(file):
    with open(file, "r") as f:
        n = int(f.readline().split()[1])
        for i in range(n):
            # this line include zero-bazed vertex index, and their index in the raw data
            # so we do not need them
            f.readline()
        m = int(f.readline().split()[1])
        row, col = [0] * m, [0] * m
        for i in range(m):
            u, v, _ = list(map(int, f.readline().split()))
            assert u != v, "contain self-loop"
            row[i], col[i] = u, v
    val = [1] * m
    A = sp.coo_matrix((val, (row, col)))
    A = A.tocsr()
    sym_err = A - A.T
    sym_check_res = np.all(np.abs(sym_err.data) < 1e-10)  # tune this value
    assert sym_check_res, 'input matrix is not symmetric!!!'
    g = dgl.DGLGraph()
    g.from_scipy_sparse_matrix(A)
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for x2dgl")
    parser.add_argument("--graph-dir", type=str, default=None, help="dir to load graphs")
    parser.add_argument("--save-file", type=str, default=None, help="file to save graphs")

    args = parser.parse_args()
    graph_dir = pathlib.Path(args.graph_dir)
    #  g = yuxiao_kdd17_graph_to_dgl("data_bin/ca-GrQc-SNAP.txtu.lpm.lscc")
    graphs = [yuxiao_kdd17_graph_to_dgl(graph_file) for graph_file in graph_dir.iterdir() if graph_file.is_file()]
    save_graphs(args.save_file, graphs)
