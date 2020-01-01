#!/bin/bash


DATE=`date +"%Y%m%d"`
python train_graph_moco.py \
 --exp $DATE \
 --model_path /data/jiezhong/graph_moco/model_kdd17 \
 --tb_path tensorboard_kdd17 \
 --model gat \
 --softmax \
 --moco \
 --readout "set2set" \
 --restart-prob 0.9 \
 --rw-hops 128 \
 --subgraph-size 64 \
 --hidden-size 64 \
 --optimizer adam \
 --weight_decay 0.0001 \
 --num-layer 6 \
 --set2set-lstm-layer 1 \
 --set2set-iter 6 \
 --num_workers 32 \
 --batch_size 32 \
 --learning_rate 0.005 \
 --num-samples 10000 \
 --nce_k 65536 \
 --gpu $1 \
