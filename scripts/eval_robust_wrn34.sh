#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/lwf_wrn34_cifar10_0.001-best"
    "trained_models/lwf_wrn34_cifar10(0.1)_0.1-best"
    "trained_models/lwf_wrn34_cifar10(0.2)_0.1-best"
    "trained_models/lwf_wrn34_cifar10(0.5)_0.1-best"
)

for model in ${MODEL_LIST[@]}; do
    echo "using $model"

    python -m exps.eval_robust_wrn34 -m ${model} -d cifar10 -n 10 --model-type wrn34 --result-file ./lwf_wrn34_cifar10.json --log lwf_wrn34_cifar10_attack.log

    valid $?
done
