#!/bin/bash
python train_graph_moco.py
 --model_path saved \
 --tb_path tensorboard \
 --num_workers 36 \
 --batch_size 64
 --rw-hops 2048 \
 --subgraph-size 64 \
 --softmax \
 --moco \
 --gpu $1 \
