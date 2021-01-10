#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_PATH=trained_models

TARGET_DOMAIN=svhntl
NUM_CLASSES=10
MODEL_ARCH=wrn28

for k in $(seq $2 $3); do
    TEACHER_MODEL=other_model

    echo "#######################################################"
    echo "simply transferring from ${TEACHER_MODEL}"
    python -m exps.transfer_learning --model=${MODEL_ARCH} --num_classes=${NUM_CLASSES} --dataset=${TARGET_DOMAIN} --teacher=${TEACHER_MODEL} -k=${k}
    valid $?

    python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/tl_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL}-last -k=${k} \
            --log=tl_${2}.log --result-file=logs/tleval_${2}.json --model-type=${MODEL_ARCH} --dataset=${TARGET_DOMAIN} --num_classes=${NUM_CLASSES}
    valid $?

done



