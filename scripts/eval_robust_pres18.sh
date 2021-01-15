#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_LIST=(
    "trained_models/sntl_1_0.4_True_pres18_mnist_5_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_5_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_True_pres18_mnist_3_at_res18_svhn-best_robust-last"
    "trained_models/sntl_1_0.4_False_pres18_mnist_3_at_res18_svhn-best_robust-last"
)

for model in ${MODEL_LIST[@]}; do
    echo "using $model"

    python -m exps.eval_robust_pres18 -m ${model} -d mnist -n 10 --model-type pres18 --result-file ./wd_sntl_pres18.json --log wd_sntl_pres18_attack.log

    valid $?
done
