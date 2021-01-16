#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/sntl_1_0.4_True_pres18_mnist_5_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_5_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_True_pres18_mnist_3_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_3_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_True_pres18_mnist_6_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_6_at_res18_svhn-best_robust-last"

    "trained_models/sntl_1_0.4_True_pres18_mnist_3_wd_fdm_True_res18_svhn_3_1.0-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_3_wd_fdm_True_res18_svhn_3_1.0-best_robust-last"
    
    "trained_models/sntl_1_0.4_True_pres18_mnist_5_wd_fdm_True_res18_svhn_5_1.0-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_5_wd_fdm_True_res18_svhn_5_1.0-best_robust-last"
    
    "trained_models/sntl_1_0.4_True_pres18_mnist_6_wd_fdm_True_res18_svhn_6_1.0-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_6_wd_fdm_True_res18_svhn_6_1.0-best_robust-last"
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