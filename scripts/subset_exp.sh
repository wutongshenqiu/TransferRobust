#!/bin/bash
source scripts/utils.sh
cuda $1

if [ -z $2 ]; then
    echo "please specify number of trainable layers"
    exit
fi
K=$2
MODEL_PATH=trained_models
ARCH=pwrn34
POWER_ITER=1
NORM_BETA=0.4
FREEZE_BN_LAYER=True

TARGET_DOMAIN=cifar10

TEACHER_MODEL_LIST=(
    at_wrn34_cifar100-best_robust
    cartl_wrn34_cifar100_4_0.01-best_robust
    cartl_wrn34_cifar100_6_0.01-best_robust
    cartl_wrn34_cifar100_8_0.01-best_robust
)

RATIO=(
    0.5
    0.2
    0.1
)

for ratio in ${RATIO[@]}; do
    for teacher in ${TEACHER_MODEL_LIST[@]}; do
        echo "#######################################################"
        echo "using teacher model ${teacher} on ${TARGET_DOMAIN}(${ratio})"

        if [ ${FREEZE_BN_LAYER} = "True" ]; then
            freeze_bn_op="--freeze-bn"
        else
            freeze_bn_op=""
        fi

        train_set="${TARGET_DOMAIN}(${ratio})"

        if [ -f ${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_${FREEZE_BN_LAYER}_${ARCH}_${train_set}_${K}_${teacher}-last ]; then
            echo "find '${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_${FREEZE_BN_LAYER}_${ARCH}_${train_set}_${K}_${teacher}-last', skip"
        else
            python -m exps.neft_spectrum_norm --model=${ARCH} --num_classes=10 --dataset=${train_set} --teacher=${teacher} -k=${K} \
                    --power-iter=${POWER_ITER} --norm-beta=${NORM_BETA} ${freeze_bn_op}
            valid $?
        fi

        # python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_${FREEZE_BN_LAYER}_${ARCH}_${train_set}_${K}_${teacher}-last -k=${K} \
        #         --model-type=${ARCH} --dataset=${TARGET_DOMAIN} --log=subset.log --result-file=logs/subset.json  --num_classes=10
        # valid $? 

    done
done