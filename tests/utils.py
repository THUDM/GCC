import torch
import argparse

from generate import main as generate_main

E2E_PATH = "saved/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999"
MOCO_PATH = "saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999"


def get_default_args():
    return argparse.Namespace(hidden_size=64, seed=0, num_shuffle=10)


def generate_emb(load_path, dataset):
    args = argparse.Namespace(
        load_path=load_path,
        dataset=dataset,
        gpu=0 if torch.cuda.is_available() else None,
    )
    generate_main(args)
