#!/bin/bash
source scripts/utils.sh

cuda $1

# TEACHER_MODEL_PATH=at_wrn28\(4\)_cifar10-best_robust
# TEACHER_MODEL_PATH=at_wrn28\(4\)_gtsrb-best_robust
TEACHER_MODEL_PATH=at_wrn34_cifar100-best_robust

TARGET_DOMAIN=gtsrb
NUM_CLASSES=43
MODEL_ARCH=$2

total_size=12630

for k in $3; do

    echo "#######################################################"
    echo "simply transferring from ${TEACHER_MODEL}"
    # python -m exps.transfer_learning --model=${MODEL_ARCH} --num_classes=${NUM_CLASSES} --dataset=${TARGET_DOMAIN} --teacher=${TEACHER_MODEL} -k=${k}
    # cli bntl -m ${MODEL_ARCH} -n ${NUM_CLASSES} -d ${TARGET_DOMAIN} -k ${k} -t ${TEACHER_MODEL_PATH}
    # valid $?

    # cli bntl -m ${MODEL_ARCH} -n ${NUM_CLASSES} -d ${TARGET_DOMAIN} -k ${k} -t ${TEACHER_MODEL_PATH} -rts
    # valid $?

    # cli bntl -m ${MODEL_ARCH} -n ${NUM_CLASSES} -d ${TARGET_DOMAIN} -k ${k} -t ${TEACHER_MODEL_PATH} -fb
    # valid $?

    cli bntl -m ${MODEL_ARCH} -n ${NUM_CLASSES} -d ${TARGET_DOMAIN} -k ${k} -t ${TEACHER_MODEL_PATH} -rts -fb
    valid $?

    # trained_model_path1=trained_models/bntl_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL_PATH}-last
    # trained_model_path2=trained_models/bntl_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL_PATH}_rts-last
    # trained_model_path3=trained_models/bntl_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL_PATH}_fb-last
    trained_model_path4=trained_models/bntl_${MODEL_ARCH}_${TARGET_DOMAIN}_${k}_${TEACHER_MODEL_PATH}_fb_rts-last

    for trained_model_path in $trained_model_path4; do
        echo ${trained_model_path}
        # python -m exps.eval_robust_wrn34 --model=${trained_model_path} -k=${k} \
        #         --log=tl_${MODEL_ARCH}_${total_size}_${TARGET_DOMAIN}_pgd20.log \
        #         --result-file=logs/tl_${MODEL_ARCH}_${total_size}_${TARGET_DOMAIN}_pgd20.json \
        #         --model-type=${MODEL_ARCH} \
        #         --dataset=${TARGET_DOMAIN} \
        #         --num_classes=${NUM_CLASSES}
        # valid $?

        python -m exps.foolbox_bench \
            --model=${trained_model_path} \
            -k=${k} \
            --log=tl_${MODEL_ARCH}_${total_size}_${TARGET_DOMAIN}_pgd100.log \
            --result-file=logs/tl_${MODEL_ARCH}_${total_size}_${TARGET_DOMAIN}_pgd100.json \
            --model-type=${MODEL_ARCH} \
            --dataset=${TARGET_DOMAIN} --num_classes=${NUM_CLASSES} --total-size ${total_size}
        valid $?
    done

done
