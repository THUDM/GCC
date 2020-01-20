gpu=$1
load_path=$2
ARGS=${@:3}

for dataset in $ARGS
do
    python test_graph_moco.py --gpu $gpu --dataset $dataset --load_path $load_path
done
