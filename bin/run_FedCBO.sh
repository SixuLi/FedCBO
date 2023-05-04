#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'



##### FedCBO for rotated MNIST dataset #####
### Train with communications
 for seed in 0
 do
 	python "${SRC}"/main.py \
 	--experiment_name "FedCBO_rotated_mnist_seed_${seed}" \
 	--data_name 'rotated_mnist' \
 	--p 4 \
 	--N 1200 \
 	--M 200 \
 	--T 100 \
 	--Lambda 1 \
 	--Sigma 0 \
 	--Alpha 10 \
 	--Gamma 0.98 \
 	--seed "$seed" \
 	--epsilon 0.5 \
 	--epsilon_decay 0.01 \
  --epsilon_threshold 0.1 \
 	--input_dim 784 \
 	--hidden_dims 200 \
 	--output_dim 10 \
 	--batch_size 128 \
  --bias \
 	--is_communication \
 	--local_epochs 10 \
 	--optimizer 'SGD' \
 	--momentum 0.9 \
 	--lr 0.1 \
 	--gpu_ids '0'
 done


### Train without communications
# for seed in {0..4}
# do
# 	python "${SRC}"/main.py \
# 	--experiment_name "LocalSGD_rotated_mnist_seed_${seed}" \
# 	--data_name 'rotated_mnist' \
# 	--p 4 \
# 	--N 1200 \
# 	--M 200 \
# 	--T 100 \
# 	--Lambda 1 \
# 	--Sigma 0 \
# 	--Alpha 1 \
# 	--Gamma 0.01 \
# 	--seed "$seed" \
# 	--input_dim 784 \
# 	--hidden_dims 200 \
# 	--output_dim 10 \
# 	--batch_size 128 \
#   --bias \
# 	--local_epochs 10 \
# 	--optimizer 'SGD' \
# 	--momentum 0.9 \
# 	--lr 0.1 \
# 	--gpu_ids '0'
# done


