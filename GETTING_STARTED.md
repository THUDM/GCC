## Getting Started with GCC

This document provides a brief intro of the usage of builtin scripts in GCC for reproducing the paper results.

TODO: For a tutorial that involves actual coding with the API,
see our [Colab Notebook]()

  - [Pretraining](#pretraining)
    - [E2E](#e2e)
    - [MoCo](#moco)
    - [Download Pretrained Models](#download-pretrained-models)
  - [Downstream Tasks](#downstream-tasks)
    - [Node Classification](#node-classification)
      - [NC Unsupervised](#nc-unsupervised)
      - [NC Supervised](#nc-supervised)
    - [Graph Classification](#graph-classification)
      - [GC Unsupervised](#gc-unsupervised)
      - [GC Supervised](#gc-supervised)
    - [Similarity Search](#similarity-search)

<!--
## How to process data

```
python x2dgl.py --graph-dir data_bin/kdd17 --save-file data_bin/dgl/graphs.bin
```
-->

### Pretraining

#### E2E

Pretrain E2E with `K = 255`:

```bash
bash scripts/pretrain.sh <gpu> --batch-size 256
```

#### MoCo

Pretrain MoCo with `K = 16384; m = 0.999`:

```bash
bash scripts/pretrain.sh <gpu> --moco --nce-k 16384
```

#### Download Pretrained Models

```bash
python scripts/download.py --url https://drive.google.com/open?id=1lYW_idy9PwSdPEC7j9IH5I5Hc7Qv-22- --path saved --fname pretrained.tar.gz
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/cabec37002a9446d9b20/?dl=1 --path saved --fname pretrained.tar.gz
```

### Downstream Tasks

Generate embeddings on multiple datasets with

```bash
bash scripts/generate.sh <gpu> <load_path> <dataset_1> <dataset_2> ...
```

For example:

```bash
bash scripts/generate.sh 0 saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False/current.pth usa_airport kdd imdb-binary
```

#### Node Classification

##### NC Unsupervised

Run baselines on multiple datasets with `bash scripts/node_classification/baseline.sh <hidden_size> <baseline:prone/graphwave> usa_airport h-index`.

Evaluate Ours on multiple datasets:

```bash
bash scripts/generate.sh <gpu> <load_path> usa_airport h-index
bash scripts/node_classification/ours.sh <load_path> <hidden_size> usa_airport h-index
```

##### NC Supervised

TODO

#### Graph Classification

##### GC Unsupervised

```bash
bash scripts/generate.sh <gpu> <load_path> imdb-binary imdb-multi collab rdt-b rdt-5k
bash scripts/graph_classification/ours.sh <load_path> <hidden_size> imdb-binary imdb-multi collab rdt-b rdt-5k
```

##### GC Supervised

Run with `bash scripts/graph_classification/finetune_ours.sh <gpu> <dataset> <epochs> <cross_validation> <load_path> (<addtional_argument_1> ...)`. For example:

Run baseline (GIN) without loading pretrained checkpoint:

```bash
bash scripts/graph_classification/finetune_ours.sh 5 rdt-b 30 False ""
```

Run baseline with degrees as input:

```bash
bash scripts/graph_classification/finetune_ours.sh 5 rdt-b 30 False "" --degree-input
```

Load pretrained checkpoint (model hyperparameters (e.g., --degree-input) will follow pretrained arguments)

```bash
bash scripts/graph_classification/finetune_ours.sh 6 rdt-b 30 False saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False/current.pth
```

#### Similarity Search

Run baselines on all datasets with `bash scripts/similarity_search/baseline.sh <hidden_size> graphwave`.

TODO: Search proper checkpoint

```bash
bash scripts/generate.sh <gpu> <load_path> kdd icdm sigir cikm sigmod icde
bash scripts/similarity_search/ours.sh <hidden_size> kdd_icdm sigir_cikm sigmod_icde
```
