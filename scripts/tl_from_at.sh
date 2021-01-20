#!/bin/bash
source scripts/utils.sh

cuda $1

TEACHER_MODEL_PATH=at_wrn34_cifar100-best_robust

TARGET_DOMAIN=gtsrb
NUM_CLASSES=43
MODEL_ARCH=wrn34

for k in 2 4 6 8 10; do

    echo "#######################################################"
    echo "simply transferring from ${TEACHER_MODEL}"
    # python -m exps.transfer_learning --model=${MODEL_ARCH} --num_classes=${NUM_CLASSES} --dataset=${TARGET_DOMAIN} --teacher=${TEACHER_MODEL} -k=${k}
    cli tl -m ${MODEL_ARCH} -n ${NUM_CLASSES} -d ${TARGET_DOMAIN} -k ${k} -t ${TEACHER_MODEL_PATH}
    valid $?

    python -m exps.eval_robust_wrn34 --model=trained_models/tl_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL_PATH}-last -k=${k} \
            --log=tl_cifar100_gtsrb.log --result-file=logs/tl_cifar100_gtsrb.json --model-type=${MODEL_ARCH} --dataset=${TARGET_DOMAIN} --num_classes=${NUM_CLASSES}
    valid $?

done



