#!/bin/bash

source scripts/utils.sh
cuda $1

# TEACHER_MODEL=wd_fdm_True_wrn34_cifar100_8_1.0-last
# TEACHER_MODEL=at_wrn34_cifar100-best_robust
TEACHER_MODEL=cartl_wrn34_cifar100_8_0.01-best_robust

MODEL_PATH=trained_models

ARCH=pwrn34
TARGET_DOMAIN=cifar10
NUM_CLASSES=10

K=8
NORM_BETA=0.4
POWER_ITER=1

FREEZE_BN="--freeze-bn"
REUSE_STATISTIC=""
REUSE_TEACHER_STATISTIC="--reuse-teacher-statistic"

TERM=`python -m exps.utils ${FREEZE_BN} ${REUSE_STATISTIC} ${REUSE_TEACHER_STATISTIC}`
valid $?

RESULT_PATH=misc_results

trained_model="sntl_${POWER_ITER}_${NORM_BETA}_${TERM}_${ARCH}_${TARGET_DOMAIN}_${K}_${TEACHER_MODEL}-last"
echo "resulting ${trained_model}"

python -m exps.neft_spectrum_norm --model=${ARCH} \
            --num_classes=${NUM_CLASSES} \
            --dataset=${TARGET_DOMAIN} \
            --teacher=${TEACHER_MODEL} \
            -k=${K} \
            --power-iter=${POWER_ITER} \
            --norm-beta=${NORM_BETA} \
            ${REUSE_STATISTIC} \
            ${REUSE_TEACHER_STATISTIC} \
            ${FREEZE_BN}
valid $?

python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/${trained_model} \
            -k=${K} \
            --model-type=${ARCH} \
            --dataset=${TARGET_DOMAIN} \
            --log=eval_${trained_model}.log \
            --result-fil=${RESULT_PATH}/explore.json \
            --num_classes=${NUM_CLASSES}
valid $?

python -m exps.auto_attack_bench --model=${MODEL_PATH}/${trained_model} \
            --dataset=${TARGET_DOMAIN} \
            --num_classes=${NUM_CLASSES} \
            --model-type=${ARCH} \
            -k=${K} \
            --log=eval_${trained_model}.log \
            --result-file=${RESULT_PATH}/explore_auto_atk.json \
            --batch-cnt=4
valid $?
    


