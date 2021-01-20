#!/bin/bash
source scripts/utils.sh

cuda $1

# teacher_model=wd_fdm_True_wrn28\(4\)_cifar10_8_1.0-best_robust
# teacher_model=at_wrn28\(4\)_cifar10-best_robust
# teacher_model=wd_fdm_True_res18_svhn_5_1.0-best_robust
# teacher_model=at_res18_svhn-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_4_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_6_1.0-best_robust
teacher_model=wd_fdm_True_wrn34_cifar100_8_1.0-best_robust

dataset=svhn
num_classes=10
model_arch=pwrn34
k=8

echo "#######################################################"
echo "neft transferring from ${teacher_model}"

echo "unfreeze bn layer"
python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
       -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4 # --reuse-teacher-statistic

echo "pgd20 attack"

unfreeze_bn_model_path=trained_models/sntl_1_0.4_False_${model_arch}_${dataset}_${k}_${teacher_model}-last

python -m exps.eval_robust_pwrn34 --model=${unfreeze_bn_model_path} -k=${k} \
        --log=new_neft_${model_arch}_${dataset}_pgd20.log --result-file=logs/new_neft_${model_arch}_${dataset}_pgd20.json \
        --model-type=${model_arch} --dataset=${dataset} --num_classes=${num_classes}
valid $?

echo "pgd100 attack"

python -m exps.foolbox_bench \
        --model=${unfreeze_bn_model_path} \
        -k=${k} \
        --log=new_neft_${model_arch}_${dataset}_pgd100.log \
        --result-file=logs/new_neft_${model_arch}_${dataset}_pgd100.json \
        --model-type=${model_arch} \
        --dataset=${dataset} --num_classes=${num_classes} --total-size 1024
valid $?
