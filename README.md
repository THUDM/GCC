# GraphMoco

## Installation

### Install PyTorch 1.3.1

### Install PyG with CUDA

https://github.com/rusty1s/pytorch_geometric/#installation

## Install DGL with CUDA

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

```
bash scripts/gin_qibin.sh
```

## Evaluation

TODO: modify `scripts/test_gin.sh` to load a pretrained model

### Node Classification

#### NC Unsupervised

TODO: Run baselines

```
bash scripts/test_node_classification.sh
```

#### NC Supervised

### Graph Classification

#### GC Unsupervised

TODO: Modify `_rwr_trace_to_dgl_graph` `data_util.py` to load entire graphs.

```
bash scripts/test_graph_classification.sh
```

#### GC Supervised

### Similarity Search

TODO: Search proper checkpoint

```
bash scripts/test_matching.sh
```

<!-- ## horovod

Install horovod with 
```
pip install horovod
```

```
horovodrun -np 4 -H localhost:4 python horovod_train_graph_moco.py --softmax --norm --moco
``` -->
