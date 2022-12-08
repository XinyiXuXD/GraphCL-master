cd /mnt/data/shared/xyxu/projects/GraphCL-master/my_TF
conda activate nlgnn
CUDA_VISIBLE_DEVICES=2 python pretrain_graphcl.py  --num_layer 5 \
--save_every 5 --embedding_dim 160 --num-group 4 \
--aug1 dropN --aug_ratio1 0.2 --aug2 permE --aug_ratio2 0.2 \
--lam_div 0.1 --epochs 100


cd /mnt/data/shared/xyxu/projects/GraphCL-master/my_TF
conda activate nlgnn
CUDA_VISIBLE_DEVICES=4 python pretrain_graphcl.py  --num_layer 5 \
--save_every 5 --embedding_dim 160 --num-group 4 \
--aug1 dropN --aug_ratio1 0.2 --aug2 permE --aug_ratio2 0.2 \
--lam_div 0.5 --epochs 100


