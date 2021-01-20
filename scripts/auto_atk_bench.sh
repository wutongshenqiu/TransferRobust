#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/sntl_1_0.4_True_wrn28(4)_svhn_8_wd_fdm_True_wrn28(4)_cifar10_8_1.0-best_robust-last"
    "trained_models/sntl_1_0.4_False_wrn28(4)_svhn_8_wd_fdm_True_wrn28(4)_cifar10_8_1.0-best_robust-last"
    
    "trained_models/sntl_1_0.4_True_wrn28(4)_svhn_8_at_wrn28(4)_cifar10-best_robust-last"
    "trained_models/sntl_1_0.4_False_wrn28(4)_svhn_8_at_wrn28(4)_cifar10-best_robust-last"
)

k_list=(8 8 8 8)
i=0

dataset=svhn
num_classes=10
model_type=wrn28\(4\)

for model in ${MODEL_LIST[@]}; do
    echo "using $model"
    python -m exps.auto_attack_bench -d=${dataset} -n=${num_classes} --model-type=${model_type} -m=${model}\
            -k=${k_list[i]} --log=cifar10_svhn_auto_atk.log --result-file=logs/cifar10_svhn_auto_atk.json --batch-cnt=4
    valid $?
    i=$((i+1))
done
