#!/bin/bash
python test_graph_moco.py \
 --model_path saved \
 --tb_path tensorboard \
 --model mpnn \
 --softmax \
 --moco \
 --readout "set2set" \
 --restart-prob 0.9 \
 --rw-hops 64 \
 --subgraph-size 64 \
 --hidden-size 32 \
 --optimizer adam \
 --num-layer 6 \
 --set2set-lstm-layer 1 \
 --set2set-iter 6 \
 --gpu $1 \
 --dataset $2 \
 --load-path saved/Grpah_MoCo0.999__dgl_softmax_16384_mpnn_layer_6_lr_0.005_decay_0.0001_bsz_32_moco_True_nce_t100_readout_set2set_subgraph_64_rw_hops_64_restart_prob_0.9_optimizer_adam_layernorm_False_s2s_lstm_layer_1_s2s_iter_6_aug_1st
