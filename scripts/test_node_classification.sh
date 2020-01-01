#!/bin/bash
gpu=$1
hidden_size=$2
test_script=${3:-scripts/test_mpnn.sh}

bash $test_script $gpu usa_airport > /dev/null
bash $test_script $gpu brazil_airport > /dev/null
bash $test_script $gpu europe_airport > /dev/null

python cogdl/scripts/train.py --task unsupervised_node_classification --dataset usa_airport --model prone from_numpy --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/usa_airport.npy
python cogdl/scripts/train.py --task unsupervised_node_classification --dataset usa_airport --model from_numpy_cat_prone --seed 0 --device-id $gpu --hidden-size $((hidden_size * 2)) --emb-path saved/usa_airport.npy
python cogdl/scripts/train.py --task unsupervised_node_classification --dataset brazil_airport --model prone from_numpy --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/brazil_airport.npy
python cogdl/scripts/train.py --task unsupervised_node_classification --dataset brazil_airport --model from_numpy_cat_prone --seed 0 --device-id $gpu --hidden-size $((hidden_size * 2)) --emb-path saved/brazil_airport.npy
python cogdl/scripts/train.py --task unsupervised_node_classification --dataset europe_airport --model prone from_numpy --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/europe_airport.npy
python cogdl/scripts/train.py --task unsupervised_node_classification --dataset europe_airport --model from_numpy_cat_prone --seed 0 --device-id $gpu --hidden-size $((hidden_size * 2)) --emb-path saved/europe_airport.npy
