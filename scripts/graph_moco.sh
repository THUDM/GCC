#!/bin/bash
python train_graph_moco.py \
 --model_path saved \
 --tb_path tensorboard \
 --softmax \
 --moco \
 --readout "avg" \
 --restart-prob 0.9 \
 --rw-hops 64 \
 --subgraph-size 64 \
 --hidden-size 32 \
 --nce_t 1\
 --layernorm \
 --gpu $1 \
