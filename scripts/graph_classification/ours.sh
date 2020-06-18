#!/bin/bash
load_path=$1
hidden_size=$2
ARGS=${@:3}

for dataset in $ARGS
do
    python gcc/tasks/graph_classification.py --dataset $dataset --hidden-size $hidden_size --model from_numpy_graph --emb-path "$load_path/$dataset.npy"
done
