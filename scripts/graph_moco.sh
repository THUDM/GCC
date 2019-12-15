#!/bin/bash
python train_graph_moco.py \
 --model_path saved \
 --tb_path tensorboard \
 --softmax \
 --moco \
 --readout "set2set" \
 --restart-prob 0.8 \
 --gpu $1 \
