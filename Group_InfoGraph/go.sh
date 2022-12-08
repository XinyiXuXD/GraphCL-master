#!/bin/bash -ex
# MUTAG, IMDB-BINARY, PROTEINS
# DD

conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/Group_InfoGraph
dataset=MUTAG
epochs=5
CUDA_VISIBLE_DEVICES=1 python InfoGraph.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'none' 'none' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax \
  --random_seed 0 1 2 3 4 --loss_emb 'IG_binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'recover' \
  --lam_div 0.6 \
  --learning_rate 0.001 \
  --num_group 3
