#!/bin/bash

source scripts/utils.sh

cuda $1

MODEL_ARCH=wrn34
BETA_NORM=0.6
NUM_CLASSES=100
TRAINSET=cifar100

python -m exps.lip_adv_train -m=${MODEL_ARCH} -n=${NUM_CLASSES} -d=${TRAINSET} --num-steps=7 --power-iter=1 --beta-norm=${BETA_NORM}
valid $?

python -m exps.eval_robust_pwrn34 --model=trained_models/snat_${MODEL_ARCH}_${TRAINSET}_1_${BETA_NORM}-last -k=1 \
                --model-type=${MODEL_ARCH} --dataset=${TRAINSET} --log=snat.log --result-file=logs/snat.json  --num_classes=${NUM_CLASSES}
valid $?