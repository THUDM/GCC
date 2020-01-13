#!/bin/bash
DATE=`date +"%Y%m%d"`

python train_graph_moco.py \
 --exp moco_64 \
 --model_path saved \
 --tb_path tensorboard \
 --model gin \
 --softmax \
 --moco \
 --readout set2set \
 --restart-prob 0.8 \
 --rw-hops 256 \
 --subgraph-size 64 \
 --hidden-size 64 \
 --optimizer adam \
 --weight_decay 0.00001 \
 --num-layer 5 \
 --set2set-lstm-layer 1 \
 --set2set-iter 6 \
 --num_workers 48 \
 --num_copies 4 \
 --batch_size 32 \
 --learning_rate 0.005 \
 --num-samples 2000 \
 --norm \
 --nce_t 0.07 \
 --save_freq 1 \
 --tb_freq 500 \
 --gpu $1 \
 --dataset ${2:-dgl}\
 --epochs 100 \
#  --finetune \
#  --resume saved/GMoCo0.999_gin_dgl_softmax_gin_layer_5_lr_0.0050_decay_0.00001_bsz_32_samples_2000_nce_t_0.07_nce_k_16384_readout_set2set_rw_hops_256_restart_prob_0.80_optimizer_adam_norm_True_s2s_lstm_layer_1_s2s_iter_6_finetune_False_seed_42_aug_1st/ckpt_epoch_10.pth
