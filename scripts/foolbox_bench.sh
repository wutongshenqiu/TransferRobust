#!/bin/bash
source scripts/utils.sh

cuda $1

dataset=mnist
model_arch=pres18
num_classes=10
MODEL_LIST=(
    "trained_models/sntl_1_0.4_False_pres18_mnist_5_wd_fdm_True_res18_svhn_5_1.0-best_robust-last"
    "trained_models/sntl_1_0.4_True_pres18_mnist_5_wd_fdm_True_res18_svhn_5_1.0-best_robust-last"
    
    "trained_models/sntl_1_0.4_False_pres18_mnist_5_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_True_pres18_mnist_5_at_res18_svhn-best_robust-last"

    "trained_models/tl_res18_mnist_5_at_res18_svhn-best_robust-last"
)


# k_list=(2 4 6 8 10)
# i=0

for model in ${MODEL_LIST[@]}; do
    echo "using $model"
    python -m exps.foolbox_bench \
            --model=${model} \
            --log=neft_${model_arch}_${dataset}_pgd100_attack.log --result-file=logs/partial_lwf_${dataset}_pgd100_attack.json --model-type=${model_arch} \
            --dataset=${dataset} --num_classes=${num_classes} --total-size 1024
    valid $?
    # i=$((i+1))
done