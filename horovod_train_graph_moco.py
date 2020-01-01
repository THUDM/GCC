#!/usr/bin/env python
# encoding: utf-8
# File Name: horovod_train_graph_moco.py
# Author: Jiezhong Qiu
# Create Time: 2020/01/01 23:59
# TODO: add horovod training

import torch
import argparse
import torch.backends.cudnn as cudnn
import horovod.torch as hvd
from graph_dataset import LoadBalanceGraphDataset, worker_init_fn
import data_util

parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--restart-prob", type=float, default=0.6)
parser.add_argument("--positional-embedding-size", type=int, default=32)
parser.add_argument("--rw-hops", type=int, default=2048)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument("--num_workers", type=int, default=4, help="num of workers to use")
args = parser.parse_args()
hvd.init()

torch.manual_seed(args.seed)

# Horovod: pin GPU to local rank.
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=1000,
            dgl_graphs_file="./data_bin/dgl/yuxiao_lscc_wo_fb_and_friendster_plus_dgl_built_in_graphs2.bin",
            num_copies=1
        )

# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1,
    collate_fn=data_util.batcher(),
    num_workers=args.num_workers,
    worker_init_fn=worker_init_fn
    )

for batch in train_loader:
    break

for batch in train_loader:
    break
