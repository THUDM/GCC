# How to process data
```
python x2dgl.py --graph-dir data_bin/kdd17 --save-file data_bin/dgl/graphs.bin
```

# How to evaluate downstream tasks

## Install CogDL

```bash
git clone https://github.com/qibinc/cogdl.git
git checkout moco
pip install -e .
```

## Export embeddings

```bash
# Modify test_gat.sh
bash scripts/test_gat.sh
```

## Use CogDL

```bash
cd /path/to/cogdl
python scripts/train.py --task unsupervised_node_classification --dataset blogcatalog --model prone from_numpy --hidden-size 64 --seed 0 --emb-path /path/to/saved/current.pt.npy
```
