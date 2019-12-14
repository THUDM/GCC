#!/bin/bash
python train_graph_moco.py \
 --model_path saved \
 --tb_path tensorboard \
 --softmax \
 --moco \
 --gpu $1 \
