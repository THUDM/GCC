#!/bin/bash
gpu=$1
hidden_size=$2
test_script=${3:-scripts/test_gin.sh}

bash $test_script $gpu imdb-binary > /dev/null
python cogdl/scripts/train.py --task graph_classification --dataset imdb-binary --model from_numpy_graph --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/imdb-binary.npy

bash $test_script $gpu imdb-multi > /dev/null
python cogdl/scripts/train.py --task graph_classification --dataset imdb-multi --model from_numpy_graph --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/imdb-multi.npy

bash $test_script $gpu collab > /dev/null
python cogdl/scripts/train.py --task graph_classification --dataset collab --model from_numpy_graph --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/collab.npy

bash $test_script $gpu rdt-b > /dev/null
python cogdl/scripts/train.py --task graph_classification --dataset rdt-b --model from_numpy_graph --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/rdt-b.npy

bash $test_script $gpu rdt-5k > /dev/null
python cogdl/scripts/train.py --task graph_classification --dataset rdt-5k --model from_numpy_graph --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path saved/rdt-5k.npy
