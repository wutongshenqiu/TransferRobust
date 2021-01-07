#!/bin/bash
source scripts/utils.sh

cuda $1

MODEL_PATH=trained_models
POWER_ITER=1
NORM_BETA=0.4
FREEZE_BN_LAYER=True

ARCH=pwrn28
TARGET_DOMAIN=svhntl

for lambda in 0.01; do
    for k in 6; do
        # TEACHER_MODEL=cartl_wrn34_cifar100_${k}_${lambda}-best_robust
        TEACHER_MODEL=other_model

        if [ ! -e "${MODEL_PATH}/${TEACHER_MODEL}" ]; then
            echo "CANNOT FIND ${MODEL_PATH}/${TEACHER_MODEL}, SKIPPING"
            continue
        fi

        if [ ${FREEZE_BN_LAYER} = "True" ]; then
            freeze_bn_op="--freeze-bn"
        else
            freeze_bn_op=""
        fi

        echo "#######################################################"
        echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"
        # python -m exps.neft_spectrum_norm --model=${ARCH} --num_classes=10 --dataset=${TARGET_DOMAIN} --teacher=${TEACHER_MODEL} -k=${k} \
        #         --power-iter=${POWER_ITER} --norm-beta=${NORM_BETA} ${freeze_bn_op}
        # valid $?

        python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_${FREEZE_BN_LAYER}_${ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL}-last -k=${k} \
                --model-type=${ARCH} --dataset=${TARGET_DOMAIN} --log=sntl.log --result-file=logs/sntl.json  --num_classes=10
        valid $?

    done
done
