#!/bin/bash
load_path=$1
hidden_size=$2
ARGS=${@:3}

for dataset in $ARGS
do
  d1=$(echo $dataset | cut -d'_' -f 1)
  d2=$(echo $dataset | cut -d'_' -f 2)
  python gcc/tasks/similarity_search.py --dataset $dataset --hidden-size $hidden_size --model from_numpy_align --emb-path-1 "$load_path/$d1.npy" --emb-path-2 "$load_path/$d2.npy"
done
