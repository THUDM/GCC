#!/bin/bash
hidden_size=$1
ARGS=${@:2}

for dataset in $ARGS
do
    python cogdl/scripts/train.py --task graph_classification --dataset $dataset --seed 0 --hidden-size $hidden_size --model from_numpy_graph --emb-path "saved/$dataset.npy"
done
