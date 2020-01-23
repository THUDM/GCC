#!/bin/bash
gpu=$1
dataset=$2
epoch=${3:-30}
cv=${4:-"False"}
load_path=${5:-""}
ARGS=${@:6}

cv_flag=''
if [[ $cv = *"True"* ]]; then
    cv_flag="--cv"
fi
resume_flag=''
if [[ $load_path != "" ]]; then
    resume_flag="--resume $load_path"
fi

python train_graph_moco.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --tb-freq 5 \
  --gpu $gpu \
  --dataset $dataset \
  --finetune \
  --epochs $epoch \
  $resume_flag \
  $cv_flag \
  $ARGS

