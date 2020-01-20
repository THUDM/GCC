#!/bin/bash
hidden_size=$1
ARGS=${@:2}

for dataset in usa_airport h-index-top-1 h-index-rand-1 h-index-rand20intop200
do
    python cogdl/scripts/train.py --task unsupervised_node_classification --dataset $dataset --seed 0 --hidden-size $hidden_size --model $ARGS
done
