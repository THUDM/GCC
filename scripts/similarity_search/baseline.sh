#!/bin/bash
hidden_size=$1
ARGS=${@:2}

for dataset in kdd_icdm sigir_cikm sigmod_icde 
do
    python cogdl/scripts/train.py --task align --dataset $dataset --seed 0 --hidden-size $hidden_size --model $ARGS
done
