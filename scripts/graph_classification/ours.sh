#!/bin/bash
gpu=$1
load_path=$2
load_name=$3
hidden_size=$4
ARGS=${@:5}

for dataset in $ARGS
do
    python test_graph_moco.py --gpu $gpu --dataset $dataset --load-path "$load_path/$load_name"
done

for dataset in $ARGS
do
    python cogdl/scripts/train.py --task graph_classification --dataset $dataset --seed 0 --hidden-size $hidden_size --model from_numpy_graph --emb-path "$load_path/$dataset.npy"
done
