#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/sntl_1_0.6_False_pwrn34_cifar10_8_at_wrn34_cifar100-best_robust-last"
    "trained_models/sntl_1_0.6_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last"
    "trained_models/sntl_1_0.4_True_pwrn34_cifar10_8_at_wrn34_cifar100-best_robust-last"
    "trained_models/sntl_1_0.4_True_pwrn34_cifar10_8_cartl_wrn34_cifar100_8_0.01-best_robust-last"
)


for atk in LinfPGD LinfDeepFool L2CW; do
    for model in ${MODEL_LIST[@]}; do
        echo "using $model"
        python -m exps.foolbox_bench -d=cifar10 -n=10 --model-type=pwrn34 -m=${model}\
                -k=8 --log=foolbox.log --result-file=logs/foolbox.json --attacker=${atk}
        valid $?
    done

    python -m exps.foolbox_bench -d=cifar10 -n=10 --model-type=wrn34 -m=trained_models/tl_wrn34_cifar10_8_at_wrn34_cifar100-best_robust-last\
            -k=8 --log=foolbox.log --result-file=logs/foolbox.json --attacker=${atk}
    valid $?
done