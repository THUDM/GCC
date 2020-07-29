#!/bin/bash
load_path=$1
gpu=$2
ARGS=${@:3}

declare -A epochs=(["usa_airport"]=30 ["h-index"]=30 ["imdb-binary"]=30 ["imdb-multi"]=30 ["collab"]=30 ["rdt-b"]=100 ["rdt-5k"]=100)

for dataset in $ARGS
do
    python train.py --exp FT --model-path saved --tb-path tensorboard --tb-freq 5 --gpu $gpu --dataset $dataset --finetune --epochs ${epochs[$dataset]} --resume "$load_path/current.pth" --cv
done
