#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/tl_wrn28(4)_gtsrb_8_at_wrn28(4)_cifar10-best_robust-last"
)

dataset=gtsrb
num_classes=43
model_arch=wrn28\(4\)
total_size=12630

for model in ${MODEL_LIST[@]}; do
    echo "using $model"

    python -m exps.foolbox_bench \
        --model=${model} \
        -k=${2} \
        --log=tl_${model_arch}_${total_size}_${dataset}_pgd100.log \
        --result-file=logs/tl_${model_arch}_${total_size}_${dataset}_pgd100.json \
        --model-type=${model_arch} \
        --dataset=${dataset} --num_classes=${num_classes} --total-size ${total_size}
    valid $?
done
