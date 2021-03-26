#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    # "trained_models/bntl_wrn34_cifar10_4_at_wrn34_cifar100-best_robust-last"
    # "trained_models/bntl_wrn34_cifar10_4_at_wrn34_cifar100-best_robust_rts-last"
    # "trained_models/bntl_wrn34_cifar10_4_at_wrn34_cifar100-best_robust_fb-last"
    # "trained_models/bntl_wrn34_cifar10_4_at_wrn34_cifar100-best_robust_fb_rts-last"

    # "trained_models/bntl_wrn34_cifar10_6_at_wrn34_cifar100-best_robust-last"
    # "trained_models/bntl_wrn34_cifar10_6_at_wrn34_cifar100-best_robust_rts-last"
    # "trained_models/bntl_wrn34_cifar10_6_at_wrn34_cifar100-best_robust_fb-last"
    # "trained_models/bntl_wrn34_cifar10_6_at_wrn34_cifar100-best_robust_fb_rts-last"

    "trained_models/bntl_wrn34_cifar10_8_at_wrn34_cifar100-best_robust-last"
    "trained_models/bntl_wrn34_cifar10_8_at_wrn34_cifar100-best_robust_rts-last"
    "trained_models/bntl_wrn34_cifar10_8_at_wrn34_cifar100-best_robust_fb-last"
    "trained_models/bntl_wrn34_cifar10_8_at_wrn34_cifar100-best_robust_fb_rts-last"
)

k_list=(4 4 4 4 6 6 6 6 8 8 8 8)
i=0

dataset=cifar10
num_classes=10
model_type=wrn34

for model in ${MODEL_LIST[@]}; do
    echo "using $model"
    python -m exps.auto_attack_bench -d=${dataset} \
        -n=${num_classes} \
        --model-type=${model_type} \
        -m=${model} \
        -k=$2 \
        --log=tl_${dataset}_{model_type}_auto_atk.log \
        --result-file=logs/tl_${dataset}_{model_type}_auto_atk.json \
        --batch-cnt=4
    valid $?
    i=$((i+1))
done
