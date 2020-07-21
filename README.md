<p align="center">
  <img src="fig.png" width="500">
  <br />
  <br />
  <a href="https://github.com/THUDM/GCC/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/THUDM/GCC" /></a>
  <a href="https://github.com/ambv/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

-------------------------------------

# GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training

Original implementation for paper [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://arxiv.org/abs/2006.09963).

GCC is a **contrastive learning** framework that implements unsupervised structural graph representation pre-training and achieves state-of-the-art on 10 datasets on 3 graph mining tasks.

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md).

## Common Issues

<details>
<summary>
"XXX file not found" when running pretraining/downstream tasks.
</summary>
<br/>
Please make sure you've downloaded the pretraining dataset or downstream task datasets according to GETTING_STARTED.md.
</details>

<details>
<summary>
Server crashes/hangs after launching pretraining experiments.
</summary>
<br/>
In addition to GPU, our pretraining stage requires a lot of computation resources, including CPU and RAM. If this happens, it usually means the CPU/RAM is exhausted on your machine. You can decrease `--num-workers` (number of dataloaders using CPU) and `--num-copies` (number of datasets copies residing in RAM). With the lowest profile, try `--num-workers 1 --num-copies 1`.

If this still fails, please upgrade your machine :). In the meanwhile, you can still download our pretrained model and evaluate it on downstream tasks.
</details>

## Citing GCC

If you use GCC in your research or wish to refer to the baseline results, please use the following BibTeX.

```
@article{qiu2020gcc,
  title={GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training},
  author={Qiu, Jiezhong and Chen, Qibin and Dong, yuxiao and Zhang, Jing and Yang, Hongxia and Ding, Ming and Wang, Kuansan and Tang, Jie},
  journal={arXiv preprint arXiv:2006.09963},
  year={2020}
}
```
