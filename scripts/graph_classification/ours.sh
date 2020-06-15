#!/bin/bash
gpu=$1
load_path=$2
hidden_size=$3
ARGS=${@:4}

for dataset in $ARGS
do
    python gcc/tasks/graph_classification.py --dataset $dataset --hidden-size $hidden_size --model from_numpy_graph --emb-path "$load_path/$dataset.npy"
done
