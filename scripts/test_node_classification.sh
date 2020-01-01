#!/bin/bash
gpu=$1
hidden_size=$2

bash scripts/test_mpnn.sh $gpu usa_airport > /dev/null

python cogdl/scripts/train.py --task unsupervised_node_classification --dataset usa_airport --model prone from_numpy --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/usa_airport.npy
python cogdl/scripts/train.py --task unsupervised_node_classification --dataset usa_airport --model from_numpy_cat_prone --seed 0 --device-id $gpu --hidden-size $((hidden_size * 2)) --emb-path saved/usa_airport.npy
