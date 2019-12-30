#!/usr/bin/env python
# encoding: utf-8
# File Name: train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 16:44
# TODO:

import argparse
import os
import time

#  import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
import torch
import dgl
import numpy as np

from graph_dataset import GraphDataset, CogDLGraphDataset, LoadBalanceGraphDataset, worker_init_fn
import data_util
from models.gcn import UnsupervisedGCN
from models.gat import UnsupervisedGAT
from models.mpnn import UnsupervisedMPNN
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss
from util import AverageMeter, adjust_learning_rate
import psutil
import warnings

try:
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb_freq", type=int, default=500, help="tb frequency")
    parser.add_argument("--save_freq", type=int, default=10, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=32, help="num of workers to use")
    parser.add_argument("--num-samples", type=int, default=10000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=60, help="number of training epochs")

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

    parser.add_argument("--amp", action="store_true", help="using mixed precision")
    parser.add_argument("--opt_level", type=str, default="O2", choices=["O1", "O2"])

    parser.add_argument("--exp", type=str, default="")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde"])

    # model definition
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "mpnn"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=2, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["root", "avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--layernorm", action="store_true", help="apply layernorm on output feats")

    # loss function
    parser.add_argument("--softmax", action="store_true", help="using softmax contrastive loss rather than NCE")
    parser.add_argument("--nce_k", type=int, default=16384)
    parser.add_argument("--nce_t", type=float, default=100)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=2048)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.6)
    parser.add_argument("--hidden-size", type=int, default=64)

    # specify folder
    parser.add_argument("--model_path", type=str, default=None, help="path to save model")
    parser.add_argument("--tb_path", type=str, default=None, help="path to tensorboard")
    parser.add_argument("--load-path", type=str, default=None, help="loading checkpoint at test time")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")

    # GPU setting
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # fmt: on
    parser.add_argument('--seed', type=int, default=42, help='random seed.')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = "softmax" if opt.softmax else "nce"

    return opt


def option_update(opt):
    prefix = "Grpah_MoCo{}".format(opt.alpha)
    opt.model_name = "{}_{}_{}_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_samples_{}_moco_{}_nce_t{}_readout_{}_rw_hops_{}_restart_prob_{}_optimizer_{}_layernorm_{}_s2s_lstm_layer_{}_s2s_iter_{}".format(
        prefix,
        opt.exp,
        opt.dataset,
        opt.method,
        opt.nce_k,
        opt.model,
        opt.num_layer,
        opt.learning_rate,
        opt.weight_decay,
        opt.batch_size,
        opt.num_samples,
        opt.moco,
        opt.nce_t,
        opt.readout,
        opt.rw_hops,
        opt.restart_prob,
        opt.optimizer,
        opt.layernorm,
        opt.set2set_lstm_layer,
        opt.set2set_iter
    )

    if opt.amp:
        opt.model_name = "{}_amp_{}".format(opt.model_name, opt.opt_level)

    opt.model_name = "{}_aug_{}".format(opt.model_name, opt.aug)

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

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch.graph_q, batch.graph_k

        graph_q_feat = graph_q.ndata["x"].cuda(opt.gpu)
        graph_k_feat = graph_k.ndata["x"].cuda(opt.gpu)

        graph_q_efeat = graph_q.edata['efeat'].cuda(opt.gpu)
        graph_k_efeat = graph_k.edata['efeat'].cuda(opt.gpu)

        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q, graph_q_feat, graph_q_efeat)
        if opt.moco:
            with torch.no_grad():
                feat_k = model_ema(graph_k, graph_k_feat, graph_k_efeat)
        else:
            # end-to-end by back-propagation (the two encoders can be different).
            feat_k = model_ema(graph_k, graph_k_feat, graph_k_efeat)

        if opt.readout == "root":
            feat_q = feat_q[batch.graph_q_roots]
            feat_k = feat_k[batch.graph_k_roots]

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        if opt.moco:
            moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "max output {out:.3f}\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    out=out[0].abs().max(),
                    mem=mem.used/1024**3
                )
            )
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            #  sw.add_scalar(
            #      "learning_rate", optimizer.param_groups[0]["lr"], global_step
            #  )
            loss_meter.reset()
            prob_meter.reset()
    return epoch_loss_meter.avg

# def main(args, trial):
def main(args):
    args = option_update(args)
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))

    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used/1024**3)
    if args.dataset == "dgl":
        train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples
        )
    else:
        train_dataset = CogDLGraphDataset(
            dataset=args.dataset,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            hidden_size=args.hidden_size,
        )
    print("setting random seeds")
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used/1024**3)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_util.batcher(),
        #  shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn
    )
    mem = psutil.virtual_memory()
    print("before training", mem.used/1024**3)

    # create model and optimizer
    n_data = train_dataset.total

    if args.model == "gcn":
        model = UnsupervisedGCN(hidden_size=args.hidden_size, num_layer=args.num_layer, readout=args.readout, layernorm=args.layernorm,
                set2set_lstm_layer=args.set2set_lstm_layer, set2set_iter=args.set2set_iter)
        model_ema = UnsupervisedGCN(
            hidden_size=args.hidden_size, num_layer=args.num_layer, readout=args.readout, layernorm=args.layernorm,
                set2set_lstm_layer=args.set2set_lstm_layer, set2set_iter=args.set2set_iter
        )
    elif args.model == "gat":
        model = UnsupervisedGAT(
                hidden_size=args.hidden_size, num_layer=args.num_layer, readout=args.readout, layernorm=args.layernorm,
                set2set_lstm_layer=args.set2set_lstm_layer, set2set_iter=args.set2set_iter
                )
        model_ema = UnsupervisedGAT(
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
        model_ema = UnsupervisedMPNN(
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

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, args.softmax
    ).cuda(args.gpu)

    assert args.softmax
    criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion = criterion.cuda(args.gpu)

    model = model.cuda(args.gpu)
    model_ema = model_ema.cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            lr_decay=args.lr_decay_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            contrast.load_state_dict(checkpoint["contrast"])
            if args.moco:
                model_ema.load_state_dict(checkpoint["model_ema"])

            if args.amp and checkpoint["opt"].amp:
                print("==> resuming amp state_dict")
                amp.load_state_dict(checkpoint["amp"])

            print(
                "=> loaded successfully '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    #  logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    sw = SummaryWriter(args.tb_folder)
    #  plots_q, plots_k = zip(*[train_dataset.getplot(i) for i in range(5)])
    #  plots_q = torch.cat(plots_q)
    #  plots_k = torch.cat(plots_k)
    #  sw.add_images('images/graph_q', plots_q, 0, dataformats="NHWC")
    #  sw.add_images('images/graph_k', plots_k, 0, dataformats="NHWC")

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
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
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
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
            if args.amp:
                state["amp"] = amp.state_dict()
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
        if args.amp:
            state["amp"] = amp.state_dict()
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

        # if (epoch + 1) % 5 == 0:
        #     trial.report(loss, epoch)
        #     if trial.should_prune():
        #         raise optuna.exceptions.TrialPruned()

    return loss


if __name__ == "__main__":

    warnings.simplefilter('once', UserWarning)
    args = parse_option()

    main(args)
    # import optuna
    # def objective(trial):
    #     args.epochs = 30
    #     args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
    #     args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    #     args.alpha = 1 - trial.suggest_loguniform('alpha', 1e-4, 1e-2)
    #     return main(args, trial)

    #, work_init_fn study = optuna.load_study(study_name='graph_moco', storage="sqlite:///example.db")
    # study.optimize(objective, n_trials=20)
