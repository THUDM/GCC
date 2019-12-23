#!/bin/bash
python train_graph_moco.py \
 --model_path /data/jiezhong/graph_moco/model_kdd17 \
 --tb_path tensorboard_kdd17 \
 --model mpnn \
 --softmax \
 --moco \
 --readout "set2set" \
 --restart-prob 0.9 \
 --rw-hops 64 \
 --subgraph-size 64 \
 --hidden-size 32 \
 --optimizer adam \
 --weight_decay 0.0001 \
 --num-layer 3 \
 --set2set-lstm-layer 1 \
 --set2set-iter 2 \
 --gpu $1 \
