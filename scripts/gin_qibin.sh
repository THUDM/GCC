#!/bin/bash
DATE=`date +"%Y%m%d"`

python train_graph_moco.py \
 --exp gin \
 --model_path saved \
 --tb_path tensorboard \
 --model gin \
 --softmax \
 --moco \
 --readout set2set \
 --restart-prob 0.8 \
 --rw-hops 256 \
 --subgraph-size 64 \
 --hidden-size 128 \
 --optimizer adam \
 --weight_decay 0.00001 \
 --num-layer 5 \
 --set2set-lstm-layer 1 \
 --set2set-iter 6 \
 --num_workers 48 \
 --num_copies 4 \
 --batch_size 32 \
 --learning_rate 0.005 \
 --num-samples 2000 \
 --norm \
 --nce_t 0.07 \
 --save_freq 1 \
 --tb_freq 500 \
 --gpu $1 \
 --dataset ${2:-dgl}\
 --epochs 100
