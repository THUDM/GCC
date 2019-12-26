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

from graph_dataset import CogDLGraphDataset, GraphDataset, batcher
from models.gat import UnsupervisedGAT
from models.gcn import UnsupervisedGCN
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
        graph_q = batch.graph_q
        bsz = graph_q.batch_size
        graph_q_feat = graph_q.ndata["x"].cuda(opt.gpu)

        with torch.no_grad():
            feat_q = model(graph_q, graph_q_feat)

        assert feat_q.shape == (bsz, opt.hidden_size)
        emb_list.append(feat_q.detach().cpu())
    return torch.cat(emb_list)

def main(args):
    args = option_update(args)

    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))

    if args.dataset == "dgl":
        train_dataset = GraphDataset(
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size,
        )
    else:
        train_dataset = CogDLGraphDataset(
            dataset=args.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size,
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

    if args.model == "gcn":
        model = UnsupervisedGCN(
            hidden_size=args.hidden_size,
            num_layer=args.num_layer,
            readout=args.readout,
            layernorm=args.layernorm,
        )
    elif args.model == "gat":
        model = UnsupervisedGAT(
                hidden_size=args.hidden_size, num_layer=args.num_layer, readout=args.readout, layernorm=args.layernorm,
                set2set_lstm_layer=args.set2set_lstm_layer, set2set_iter=args.set2set_iter
                )
    elif args.model == "mpnn":
        model = UnsupervisedMPNN(
                node_input_dim=args.hidden_size,
                edge_input_class=8,
                edge_input_dim=args.hidden_size,
                output_dim=args.hidden_size,
                node_hidden_dim=args.hidden_size,
                edge_hidden_dim=args.hidden_size,
                num_step_message_passing=args.num_layer,
                num_step_set2set=args.set2set_iter,
                num_layer_set2set=args.set2set_lstm_layer
                )
    else:
        raise NotImplementedError("model not supported {}".format(args.model))

    model = model.cuda(args.gpu)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    model_path = os.path.join(args.model_folder, "current.pth")
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        print(
            "=> loaded successfully '{}' (epoch {})".format(
                model_path, checkpoint["epoch"]
            )
        )
        del checkpoint
        torch.cuda.empty_cache()

        emb = test_moco(train_loader, model, args)
        np.save(model_path[:args.resume.find(".pth")], emb.numpy())
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

if __name__ == "__main__":
    args = parse_option()
    main(args)
