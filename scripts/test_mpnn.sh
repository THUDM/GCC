#!/bin/bash
python test_graph_moco.py \
 --model_path saved \
 --tb_path tensorboard \
 --model mpnn \
 --softmax \
 --moco \
 --readout "set2set" \
 --restart-prob 0.9 \
 --rw-hops 64 \
 --subgraph-size 64 \
 --hidden-size 32 \
 --optimizer adam \
 --num-layer 6 \
 --set2set-lstm-layer 1 \
 --set2set-iter 6 \
 --gpu $1 \
 --dataset $2 \
