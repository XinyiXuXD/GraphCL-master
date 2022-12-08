#!/bin/bash -ex
# DD, IMDB-BINARY, PROTEINS

conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU_IMDB_DD_PROTEINS
dataset=IMDB-BINARY
epochs=5
CUDA_VISIBLE_DEVICES=4 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'none' 'none' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'test' --pre_transform true --bias false \
  --lam_div 0.5 \
  --learning_rate 0.0001 \
  --num_group 4

CUDA_VISIBLE_DEVICES=4 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'none' 'none' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'mul_linear' --pre_transform true --bias false \
  --lam_div 0.5 \
  --learning_rate 0.0001 \
  --num_group 4 --grouping-layer 'mul_linear'



conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU_IMDB_DD_PROTEINS
dataset=DD
epochs=15
CUDA_VISIBLE_DEVICES=4 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'none' 'none' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'best' --pre_transform false \
  --lam_div 0.5 \
  --learning_rate 0.0001 \
  --num_group 4 --grouping-layer 'query'


conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU_IMDB_DD_PROTEINS
dataset=PROTEINS
epochs=10
CUDA_VISIBLE_DEVICES=3 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'dnodes' 'dnodes' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'best' --pre_transform true \
  --lam_div 1.1 \
  --learning_rate 0.001 \
  --num_group 4 --grouping-layer 'query'


CUDA_VISIBLE_DEVICES=3 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'dnodes' 'dnodes' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'mul_linear' --pre_transform true \
  --lam_div 1.1 \
  --learning_rate 0.001 \
  --num_group 4 --grouping-layer 'mul_linear'


conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU_IMDB_DD_PROTEINS
dataset=IMDB-BINARY
epochs=5
CUDA_VISIBLE_DEVICES=5 python gsimclr_club.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'none' 'none' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_club' --batch-size 128 \
  --folder_name 'div_club' --pre_transform true \
  --lam_div 0.5  --learning_rate 0.0001  --num_group 4 \
  --club_type CLUBSample --mi_epochs 5 --club_hidden 40 \
  --club_fea_norm true


conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU_IMDB_DD_PROTEINS
dataset=DD
epochs=15
CUDA_VISIBLE_DEVICES=6 python gsimclr_club.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'none' 'none' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_club' --batch-size 128 \
  --folder_name 'div_club' --pre_transform true \
  --lam_div 0.5  --learning_rate 0.0001  --num_group 4 \
  --club_type CLUBSample --mi_epochs 3 --club_hidden 40


conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU_IMDB_DD_PROTEINS
dataset=PROTEINS
epochs=10
CUDA_VISIBLE_DEVICES=7 python gsimclr_club.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'dnodes' 'dnodes' --aug_ratio 0.1 0.1 --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_club' --batch-size 128 \
  --folder_name 'div_club' --pre_transform true \
  --lam_div 0.5  --learning_rate 0.001  --num_group 5 \
  --club_type CLUBSample --mi_epochs 3 --club_hidden 40


