## Installation

### Requirements

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.4.0
- [DGL with CUDA](https://www.dgl.ai/pages/start.html)
  - Install [RDKit](conda install -c conda-forge rdkit) with `conda install -c conda-forge rdkit`.
- `pip install -r requirements.txt`

### Datasets

#### Pre-training datasets

```bash
python scripts/download.py --url https://drive.google.com/open?id=1JCHm39rf7HAJSp-1755wa32ToHCn2Twz --path data --fname small.bin
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/b37eed70207c468ba367/?dl=1 --fname small.bin
```

#### Downstream datasets

```bash
python scripts/download.py --url https://drive.google.com/open?id=12kmPV3XjVufxbIVNx5BQr-CFM9SmaFvM --path data --fname downstream.tar.gz
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/2535437e896c4b73b6bb/?dl=1 --fname downstream.tar.gz
```

### Common Installation Issues
