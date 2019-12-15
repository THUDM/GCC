#!/bin/bash
python train_graph_moco.py \
 --model_path /data/jiezhong/graph_moco/model \
 --tb_path tensorboard \
 --softmax \
 --moco \
 --readout "set2set" \
 --restart-prob 0.9 \
 --rw-hops 64 \
 --subgraph-size 64 \
 --hidden-size 32 \
 --gpu $1 \
