#!/bin/bash
python train_graph_moco.py \
 --model_path saved \
 --tb_path tensorboard \
 --softmax \
 --moco \
 --optimizer adam \
 --hidden-size 32 \
 --model mpnn \
 --num-layer 6 \
 --batch_size 32 \
 --readout "set2set" \
 --set2set-lstm-layer 1 \
 --set2set-iter 6 \
 --rw-hops 64 \
 --restart-prob 0.9 \
 --subgraph-size 64 \
 --gpu $1 \
 --dataset $2 \
