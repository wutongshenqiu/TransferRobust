#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/sntl_1_0.6_False_pwrn34_cifar10_8_at_wrn34_cifar100-best_robust-last"
    "trained_models/sntl_1_0.6_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last"
    "trained_models/sntl_1_0.6_True_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last"
    "trained_models/sntl_1_0.6_True_pwrn34_cifar10_8_fm_fdm_wrn34_cifar100_8_0.01-last-last"
    "trained_models/sntl_1_0.4_True_pwrn34_cifar10_8_at_wrn34_cifar100-best_robust-last"
    "trained_models/sntl_1_0.4_True_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last"
)

for model in ${MODEL_LIST[@]}; do
    echo "using $model"
    python -m exps.auto_attack_bench -d=cifar10 -n=10 --model-type=pwrn34 -m=${model}\
            -k=8 --log=auto_atk.log --result-file=logs/auto_atk.json --batch-cnt=4
    valid $?
done

python -m exps.auto_attack_bench -d=cifar10 -n=10 --model-type=wrn34 -m=trained_models/tl_wrn34_cifar10_8_at_wrn34_cifar100-best_robust-last\
        -k=8 --log=auto_atk.log --result-file=logs/auto_atk.json --batch-cnt=4 
valid $?