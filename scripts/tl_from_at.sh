#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_PATH=trained_models

for k in $(seq $2 $3); do
    TEACHER_MODEL=at_wrn34_cifar100-best_robust

    echo "#######################################################"
    echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"
    python -m exps.transfer_learning --model=wrn34 --num_classes=10 --dataset=svhntl --teacher=${TEACHER_MODEL} -k=${k}
    valid $?

    python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/tl_wrn34_svhntl_${k}_${TEACHER_MODEL}-last -k=${k} \
            --log=tl_${2}.log --result-file=logs/tleval_${2}.json --model-type=wrn34 --dataset=svhntl --num_classes=10
    valid $?

done



