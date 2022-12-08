#!/bin/bash -ex
# MUTAG (188), IMDB-BINARY (1000), PROTEINS (1113),  DD(1178)
# REDDIT-BINARY (2000), NCI1(4110), REDDIT-MULTI-5K (4999), COLLAB (5000)


conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU
dataset=MUTAG
epochs=5
CUDA_VISIBLE_DEVICES=6 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'dnodes' 'dnodes'  --epochs $epochs   --att-norm softmax \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_club_bd' --batch-size 128 \
  --folder_name 'div_club_bd' \
  --lam_div 0.5 0.1 0.3 0.7 0.9 \
  --learning_rate 0.001 \
  --num_group 4 --embedding_dim 160


conda activate py3.7
cd /data/xxy/GraphCL-master/unsupervised_TU
dataset=REDDIT-BINARY
epochs=5
CUDA_VISIBLE_DEVICES=1 python gsimclr1.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'dnodes' 'dnodes' --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_bd' --batch-size 128 \
  --folder_name 'mul_linear' \
  --lam_div 0.5 \
  --learning_rate 0.001 \
  --num_group 4 --grouping-layer 'mul_linear'


conda activate nlgnn
cd /mnt/data/shared/xyxu/projects/GraphCL-master/unsupervised_TU
dataset=COLLAB
epochs=5
CUDA_VISIBLE_DEVICES=7 python gsimclr_club.py --DS $dataset  --local --num-gc-layers 3 --log-interval 1 \
  --aug 'dnodes' 'dnodes'  --epochs $epochs   --att-norm softmax   --reduction_ratio 1 \
  --random_seed 0 1 2 3 4 --loss_emb 'binomial_deviance' --loss_div 'div_club' --batch-size 128 \
  --folder_name 'div_club' --pre_transform true \
  --lam_div 0.5  --learning_rate 0.0001  --num_group 4 \
  --club_type CLUBSample --mi_epochs 1 --club_hidden 40




