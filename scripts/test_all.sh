#!/bin/bash
gpu=$1
hidden_size=$2
test_script=${3:-scripts/test_mpnn.sh}

source scripts/test_node_classification.sh
source scripts/test_matching.sh
