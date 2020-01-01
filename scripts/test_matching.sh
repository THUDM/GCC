#!/bin/bash
gpu=$1
hidden_size=$2

bash scripts/test_mpnn.sh $gpu kdd > /dev/null
bash scripts/test_mpnn.sh $gpu icdm > /dev/null
bash scripts/test_mpnn.sh $gpu sigir > /dev/null
bash scripts/test_mpnn.sh $gpu cikm > /dev/null
bash scripts/test_mpnn.sh $gpu sigmod > /dev/null
bash scripts/test_mpnn.sh $gpu icde > /dev/null

python cogdl/scripts/train.py --task align --dataset kdd_icdm --model from_numpy_align --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path-1 saved/kdd.npy --emb-path-2 saved/icdm.npy
python cogdl/scripts/train.py --task align --dataset sigir_cikm --model from_numpy_align --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path-1 saved/sigir.npy --emb-path-2 saved/cikm.npy
python cogdl/scripts/train.py --task align --dataset sigmod_icde --model from_numpy_align --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path-1 saved/sigmod.npy --emb-path-2 saved/icde.npy
