#!/bin/bash

source scripts/utils.sh

cuda $1

python -m exps.lip_adv_train -m=wrn34 -n=100 -d=cifar100 --num-steps=7
valid $?

python -m exps.eval_robust_pwrn34 --model=trained_models/snat_wrn34_cifar100_1_1.0-last -k=1 \
                --model-type=wrn34 --dataset=cifar100 --log=snat.log --result-file=logs/snat.json  --num_classes=100
valid $?