#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import os
import time

import numpy as np
import tensorboard_logger as tb_logger
import torch

from data_util import batcher
from graph_dataset import CogDLGraphDataset, GraphDataset
from models.gat import UnsupervisedGAT
from models.gcn import UnsupervisedGCN
from models.graph_encoder import GraphEncoder
from models.mpnn import UnsupervisedMPNN
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss
from train_graph_moco import option_update, parse_option
from util import AverageMeter, adjust_learning_rate


def test_moco(train_loader, model, opt):
    """
    one epoch training for moco
    """

    model.eval()

    emb_list = []
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q.to(torch.device(opt.gpu))
        graph_k.to(torch.device(opt.gpu))

        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)

        assert feat_q.shape == (bsz, opt.hidden_size)
        # emb_list.append(feat_q.detach().cpu())
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


def main(args):
    parser = argparse.ArgumentParser("argument for training")
    # fmt: off
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde"])
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    args_test = parser.parse_args()

    if os.path.isfile(args_test.load_path):
        print("=> loading checkpoint '{}'".format(args_test.load_path))
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print(
            "=> loaded successfully '{}' (epoch {})".format(
                args_test.load_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]

    assert args_test.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args_test.gpu))

    if args_test.dataset == "dgl":
        train_dataset = GraphDataset(
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size,
        )
    else:
        train_dataset = CogDLGraphDataset(
            dataset=args_test.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create model and optimizer
    n_data = len(train_dataset)

    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq,
        max_edge_freq=args.max_edge_freq,
        freq_embedding_size=args.freq_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        num_step_set2set=args.set2set_iter,
        num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model,
        norm=args.norm
    )

    model = model.cuda(args_test.gpu)

    model.load_state_dict(checkpoint["model"])

    del checkpoint
    torch.cuda.empty_cache()

    args.gpu = args_test.gpu
    emb = test_moco(train_loader, model, args)
    np.save(os.path.join(args.model_path, args_test.dataset), emb.numpy())


if __name__ == "__main__":
    args = parse_option()
    main(args)
