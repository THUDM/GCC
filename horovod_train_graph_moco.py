#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import os
import time
import torch.backends.cudnn as cudnn
import horovod.torch as hvd
from torch.utils.tensorboard import SummaryWriter
import torch
import dgl
import numpy as np
from graph_dataset import CogDLGraphDataset, LoadBalanceGraphDataset, worker_init_fn
import data_util
from models.graph_encoder import GraphEncoder
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss
from util import HorovodAverageMeter
import psutil
import warnings

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    #  parser.add_argument("--tb_freq", type=int, default=500, help="tb frequency")
    parser.add_argument("--tb_freq", type=int, default=1, help="tb frequency")
    parser.add_argument("--save_freq", type=int, default=10, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--max-node-per-batch", type=int, default=1024, help="dynamic batching")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--num_copies", type=int, default=1, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=10000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=60, help="number of training epochs")
    parser.add_argument("--dgl-graphs-file", type=str,
            default="./data_bin/dgl/yuxiao_lscc_wo_fb_and_friendster_plus_dgl_built_in_graphs.bin",
            help="dgl graphs to pretrain")

    # optimization
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])


    parser.add_argument("--exp", type=str, default="horovod")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde"])

    # model definition
    parser.add_argument("--model", type=str, default="mpnn", choices=["gat", "mpnn"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=2, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["root", "avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", help="apply 2-norm on output feats")

    # loss function
    parser.add_argument("--softmax", action="store_true", help="using softmax contrastive loss rather than NCE")
    parser.add_argument("--nce_k", type=int, default=16384)
    parser.add_argument("--nce_t", type=float, default=100)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.9)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--freq-embedding-size", type=int, default=16)

    # specify folder
    parser.add_argument("--model_path", type=str, default="/data/jiezhong/graph_moco/model_kdd17", help="path to save model")
    parser.add_argument("--tb_path", type=str, default="./tensorboard_kdd17", help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    parser.add_argument('--seed', type=int, default=42, help='random seed.')

    # horovod
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                            'executing allreduce across workers; it multiplies '
                            'total batch size.')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')

    opt = parser.parse_args()
    assert opt.positional_embedding_size % 2 == 0
    opt.allreduce_batch_size = opt.batch_size * opt.batches_per_allreduce

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = "softmax" if opt.softmax else "nce"

    return opt


def option_update(opt):
    prefix = "GMoCo{}".format(opt.alpha)
    opt.model_name = "{}_{}_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_samples_{}_nce_t_{}_nce_k_{}_readout_{}_rw_hops_{}_restart_prob_{}_optimizer_{}_norm_{}_s2s_lstm_layer_{}_s2s_iter_{}".format(
        prefix,
        opt.exp,
        opt.dataset,
        opt.method,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.num_samples,
        opt.nce_t,
        opt.nce_k,
        opt.readout,
        opt.rw_hops,
        opt.restart_prob,
        opt.optimizer,
        opt.norm,
        opt.set2set_lstm_layer,
        opt.set2set_iter
    )

    opt.model_name = "{}_aug_{}".format(opt.model_name, opt.aug)

    opt.verbose = 1 if hvd.rank() == 0 else 0

    if opt.verbose:
        if opt.load_path is None:
            opt.model_folder = os.path.join(opt.model_path, opt.model_name)
            if not os.path.isdir(opt.model_folder):
                os.makedirs(opt.model_folder)
        else:
            opt.model_folder = opt.load_path

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)
    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_moco(
    epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt
):
    """
    one epoch training for moco
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()
    device = torch.device(torch.cuda.current_device())

    loss_meter = HorovodAverageMeter("moco_loss")
    prob_meter = HorovodAverageMeter('moco_prob')
    graph_size = HorovodAverageMeter("graph_size")
    batch_size = HorovodAverageMeter("batch_size")

    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch

        graph_q.to(device)
        graph_k.to(device)

        bsz = graph_q.batch_size

        # ===================forward=====================
        feat_q = model(graph_q)
        if opt.moco:
            with torch.no_grad():
                feat_k = model_ema(graph_k)
        else:
            # end-to-end by back-propagation (the two encoders can be different).
            feat_k = model_ema(graph_k)

        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss)
        prob_meter.update(prob)
        graph_size.update(
                torch.tensor((graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz)
                )
        batch_size.update(
                torch.tensor(float(bsz))
                )

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)


        # tensorboard logger
        if sw and (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("batch_size", batch_size.avg, global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            batch_size.reset()

def main(args):
    hvd.init()
    args = option_update(args)
    if args.verbose:
        print(args)

    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(hvd.local_rank())
    cudnn.benchmark = True

    # Horovod: write TensorBoard logs on first worker.
    sw = SummaryWriter(args.tb_folder) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(args.num_workers + 1) # do we need +1?

    if args.verbose:
        mem = psutil.virtual_memory()
        print("before construct dataset", mem.used/1024**3)
    if args.dataset == "dgl":
        train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            dgl_graphs_file=args.dgl_graphs_file,
            num_copies=1,
        )
    else:
        train_dataset = CogDLGraphDataset(
            dataset=args.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )

    if args.verbose:
        mem = psutil.virtual_memory()
        print("before construct dataloader", mem.used/1024**3)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_util.dynamic_batcher(args.max_node_per_batch),
        #  shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn
    )
    if args.verbose:
        mem = psutil.virtual_memory()
        print("before training", mem.used/1024**3)

    # create model and optimizer
    n_data = train_dataset.total

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
            norm=args.norm,
            gnn_model=args.model
            )
    model_ema = GraphEncoder(
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
            norm=args.norm,
            gnn_model=args.model
            )

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, args.softmax
    ).cuda()

    assert args.softmax
    criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion = criterion.cuda()

    model = model.cuda()
    model_ema = model_ema.cuda()

    lr_scaler = args.batches_per_allreduce * hvd.size()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate * lr_scaler,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate * lr_scaler,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate * lr_scaler,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce
        )

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_parameters(model_ema.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # routine
    for epoch in range(1, args.epochs + 1):

        #  adjust_learning_rate(epoch, args, optimizer)

        if args.verbose:
            print("==> training...")

        time1 = time.time()
        loss = train_moco(
            epoch,
            train_loader,
            model,
            model_ema,
            contrast,
            criterion,
            optimizer,
            sw,
            args
        )
        time2 = time.time()
        if args.verbose:
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if args.verbose and epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {
                "opt": args,
                "model": model.state_dict(),
                "contrast": contrast.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            if args.moco:
                state["model_ema"] = model_ema.state_dict()
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print("==> Saving...")
        state = {
            "opt": args,
            "model": model.state_dict(),
            "contrast": contrast.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if args.moco:
            state["model_ema"] = model_ema.state_dict()
        save_file = os.path.join(args.model_folder, "current.pth")
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()


    return loss


if __name__ == "__main__":

    warnings.simplefilter('once', UserWarning)
    args = parse_option()

    main(args)

