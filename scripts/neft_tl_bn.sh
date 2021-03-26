#!/bin/bash
source scripts/utils.sh

cuda $1

# teacher_model=wd_fdm_True_wrn28\(4\)_cifar10_8_1.0-best_robust
teacher_model=at_wrn28\(4\)_cifar10-best_robust
# teacher_model=wd_fdm_True_res18_svhn_5_1.0-best_robust
# teacher_model=at_res18_svhn-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_4_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_6_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_8_1.0-best_robust
# teacher_model=new_wd_fdm_False_wrn28-4_cifar10_8_1.0-last

dataset=$2
num_classes=$3
model_arch=pwrn28\(4\)
k=$4

echo "#######################################################"
echo "neft transferring from ${teacher_model}"

echo "unfreeze bn layer"
python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
       -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4  --reuse-teacher-statistic


unfreeze_bn_model_path=trained_models/sntl_1_0.4_rts_False_${model_arch}_${dataset}_${k}_${teacher_model}-last

# echo "pgd20 attack"
# python -m exps.eval_robust_pwrn34 --model=${unfreeze_bn_model_path} -k=${k} \
#         --log=rts_new_neft_${model_arch}_${dataset}_pgd20.log --result-file=logs/rts_new_neft_${model_arch}_${dataset}_pgd20.json \
#         --model-type=${model_arch} --dataset=${dataset} --num_classes=${num_classes}
# valid $?

echo "pgd100 attack"

python -m exps.foolbox_bench \
        --model=${unfreeze_bn_model_path} \
        -k=${k} \
        --log=rts_new_neft_${model_arch}_${dataset}_pgd100.log \
        --result-file=logs/rts_new_neft_${model_arch}_${dataset}_pgd100.json \
        --model-type=${model_arch} \
        --dataset=${dataset} --num_classes=${num_classes} --total-size 1024
valid $?
