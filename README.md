# GraphMoco

## Installation

### Install PyTorch 1.3.1

### Install PyG with CUDA

https://github.com/rusty1s/pytorch_geometric/#installation

### Install DGL with CUDA

https://www.dgl.ai/pages/start.html

### Install CogDL

```bash
git submodule init
git submodule update
cd cogdl
pip install -e .
```

## How to process data

```
python x2dgl.py --graph-dir data_bin/kdd17 --save-file data_bin/dgl/graphs.bin
```

## Pretraining

Run negative sampling (number negative samples = batch_size - 1, default to 32)

```
bash scripts/pretrain.sh <gpu>
```

Run negative sampling with larger batch size (and negative sample size)

```
bash scripts/pretrain.sh <gpu> --batch-size 256
```

Run moco with queue size = 32:

```
bash scripts/pretrain.sh <gpu> --moco
```

Run moco with larger queue size:

```
bash scripts/pretrain.sh <gpu> --moco --nce-k 256
```

Run with larger model hidden size:

```
bash scripts/pretrain.sh ... --hidden-size 512
```

Run with degrees as input features:

```
bash scripts/pretrain.sh ... --degree-input
```

## Downstream Tasks

Generate embeddings on multiple datasets with

```bash
bash scripts/generate.sh <gpu> <load_path> <dataset_1> <dataset_2> ...
```

TODO: Modify `_rwr_trace_to_dgl_graph` in `data_util.py` to load entire graphs.

For example:

```bash
bash scripts/generate.sh 0 saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False/current.pth usa_airport kdd imdb-binary
```

### Node Classification

#### NC Unsupervised

Run baselines on all datasets with `bash scripts/node_classification/baseline.sh <hidden_size> prone graphwave`.

Run ours on multiple datasets:

```bash
bash scripts/generate.sh <gpu> <load_path> usa_airport h-index-top-1 h-index-rand-1 h-index-rand20intop200
bash scripts/node_classification/ours.sh <hidden_size> usa_airport h-index-top-1 h-index-rand-1 h-index-rand20intop200
```

#### NC Supervised

TODO

### Graph Classification

#### GC Unsupervised

```bash
bash scripts/generate.sh <gpu> <load_path> imdb-binary imdb-multi collab rdt-b rdt-5k
bash scripts/graph_classification/ours.sh <hidden_size> imdb-binary imdb-multi collab rdt-b rdt-5k
```

#### GC Supervised

TODO

### Similarity Search

Run baselines on all datasets with `bash scripts/similarity_search/baseline.sh <hidden_size> graphwave`.

TODO: Search proper checkpoint

```bash
bash scripts/generate.sh <gpu> <load_path> kdd icdm sigir cikm sigmod icde
bash scripts/similarity_search/ours.sh <hidden_size> kdd_icdm sigir_cikm sigmod_icde
```
