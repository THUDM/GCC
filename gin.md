# Trying to reproduce GIN

send to: keyulu@mit.edu; weihuahu@stanford.edu; jure@cs.stanford.edu; stefje@mit.edu

cc: chenqibin422@gmail.com jietang@tsinghua.edu.cn yuxdong@microsoft.com

Dear Authors of GIN:

This is Jiezhong Qiu from Tsinghua University. We are working on a Graph Neural Network paper and want to use GIN model as our baseline. We wonder if you would like to share some insights on training GIN model.

We download the code from https://github.com/weihua916/powerful-gnns, the official implementation. To reproduce the paper results, we run commands according to README and information provided by the authors in repo issues:

We run the following two settings, as suggested in the paper.

1. Default setting: 
```bash
python main.py --dataset IMDBBINARY --device 0 --fold-idx 0 > imdb-b_0.txt
python main.py --dataset IMDBMULTI --device 0 --fold-idx 0 > imdb-m_0.txt
python main.py --dataset COLLAB --device 0 --fold-idx 0 > collab_0.txt
python main.py --dataset REDDITBINARY --device 0 --fold-idx 0 > rdt-b_0.txt
python main.py --dataset REDDITMULTI5K --device 0 --fold-idx 0 > rdt-5k_0.txt
```
2. Turn on `degree_as_tag`:
```bash
python main.py --dataset IMDBBINARY --degree_as_tag --device 0 --fold-idx 0 > imdb-b_0.txt
python main.py --dataset IMDBMULTI --degree_as_tag --device 0 --fold-idx 0 > imdb-m_0.txt
python main.py --dataset COLLAB --degree_as_tag --device 0 --fold-idx 0 > collab_0.txt
python main.py --dataset REDDITBINARY --degree_as_tag --device 0 --fold-idx 0 > rdt-b_0.txt
python main.py --dataset REDDITMULTI5K --degree_as_tag --device 0 --fold-idx 0 > rdt-5k_0.txt
```

We repeated each command for 3 times with `--fold-idx 0`, `--fold-idx 1`, `--fold-idx 2` and obtained the following results:

| Dataset         | IMDB-B     | IMDB-M     | RDT-B      | RDT-M5K    | COLLAB     |
|-----------------|------------|------------|------------|------------|------------|
| Reported        | 75.1 ± 5.1 | 52.3 ± 2.8 | 92.4 ± 2.5 | 57.5 ± 1.5 | 80.2 ± 1.9 |
| Reproduced      | 73.3 ± 2.4 | 49.8 ± 1.3 | 74.9 ± 1.0 | 52.4 ± 2.4 | 63.9 ± 1.0 |
| + degree_as_tag | 74.3 ± 4.0 | 48.2 ± 1.9 | 76.3 ± 2.3 | 44.1 ± 1.9 | 78.9 ± 1.0 |

In a recent github repo issue (https://github.com/weihua916/powerful-gnns/issues/8), one of the authors sugeested tune hyperparmeters and select epoch that achieved the maximum averaged 10-fold validation accuracy.

We wonder if you could kindly share:

1. the preferred hyperparameters
2. scripts for evaluation, e.g. code for selecting the best epoch on the 10-fold as mentioned in the paper.
3. any specific software and hardware requirements, e.g., PyTorch, CUDA version that could cause such difference in experimental results.

that might greatly help us to produce the reported results in the GIN paper.

We want to thank you again for the excellent model and look forward to hearing from you.

Best,
Jiezhong
