# How to process data
```
python x2dgl.py --graph-dir data_bin/kdd17 --save-file data_bin/dgl/graphs.bin
```

# How to evaluate downstream tasks

## Install CogDL

```bash
git submodule init
git submodule update
cd cogdl
pip install -e .
```

## Use CogDL

See `test_node_classification.sh` and `test_matching.sh`
