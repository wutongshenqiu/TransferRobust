#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/lwf_wrn34_cifar10(0.1)_0.1-best"
    "trained_models/lwf_wrn34_cifar10(0.2)_0.1-best"
    "trained_models/lwf_wrn34_cifar10(0.5)_0.1-best"
)

dataset=cifar10
num_classes=10
model_arch=wrn34
total_size=10000

for model in ${MODEL_LIST[@]}; do
    echo "using $model"

    python -m exps.foolbox_bench \
        --model=${model} \
        -k=${2} \
        --log=tl_${model_arch}_${total_size}_${dataset}_pgd100.log \
        --result-file=logs/lwf_${model_arch}_${total_size}_${dataset}_pgd100.json \
        --model-type=${model_arch} \
        --dataset=${dataset} --num_classes=${num_classes} --total-size ${total_size}
    valid $?
done
