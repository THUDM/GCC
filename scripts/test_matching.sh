#!/bin/bash
gpu=$1
hidden_size=$2
test_script=${3:-scripts/test_mpnn.sh}

bash $test_script $gpu kdd > /dev/null
bash $test_script $gpu icdm > /dev/null
bash $test_script $gpu sigir > /dev/null
bash $test_script $gpu cikm > /dev/null
bash $test_script $gpu sigmod > /dev/null
bash $test_script $gpu icde > /dev/null

python cogdl/scripts/train.py --task align --dataset kdd_icdm --model from_numpy_align --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path-1 saved/kdd.npy --emb-path-2 saved/icdm.npy
python cogdl/scripts/train.py --task align --dataset sigir_cikm --model from_numpy_align --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path-1 saved/sigir.npy --emb-path-2 saved/cikm.npy
python cogdl/scripts/train.py --task align --dataset sigmod_icde --model from_numpy_align --seed 0 --device-id $gpu --hidden-size $hidden_size --emb-path-1 saved/sigmod.npy --emb-path-2 saved/icde.npy
