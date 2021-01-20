#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/sntl_1_0.4_True_wrn28(4)_svhn_8_wd_fdm_True_wrn28(4)_cifar10_8_1.0-best_robust-last"
    "trained_models/sntl_1_0.4_False_wrn28(4)_svhn_8_wd_fdm_True_wrn28(4)_cifar10_8_1.0-best_robust-last"
    
    "trained_models/sntl_1_0.4_True_wrn28(4)_svhn_8_at_wrn28(4)_cifar10-best_robust-last"
    "trained_models/sntl_1_0.4_False_wrn28(4)_svhn_8_at_wrn28(4)_cifar10-best_robust-last"
)

dataset=svhn
num_classes=10
model_arch=wrn28\(4\)

for model in ${MODEL_LIST[@]}; do
    echo "using $model"

    python -m exps.eval_robust_wrn34 -m ${model} -d ${dataset} \
           -n ${num_classes} --model-type ${model_arch} 
           --result-file logs/cifar10_svhn_pgd20.json --log cifar10_svhn_pgd20.log

    valid $?
done
