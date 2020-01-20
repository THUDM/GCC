#!/bin/bash
hidden_size=$1
ARGS=${@:2}

for dataset in $ARGS
do
  d1=$(echo $dataset | cut -d'_' -f 1)
  d2=$(echo $dataset | cut -d'_' -f 2)
  python cogdl/scripts/train.py --task align --dataset $dataset --seed 0 --hidden-size $hidden_size --model from_numpy_align --emb-path-1 "saved/$d1.npy" --emb-path-2 "saved/$d2.npy"
done
