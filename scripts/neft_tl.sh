#!/bin/bash
source scripts/utils.sh

cuda $1

# teacher_model=wd_fdm_True_wrn28\(4\)_cifar10_8_1.0-best_robust
# teacher_model=at_wrn28\(4\)_cifar10-best_robust
# teacher_model=cartl_wrn28\(4\)_cifar10_8_0.01-best_robust
teacher_model=cartl_wrn28\(4\)_cifar10_6_0.01-best_robust
# teacher_model=cartl_wrn28-4_cifar10_6_0.01-last
# teacher_model=cartl_wrn28\(4\)_cifar10_6_0.005-best_robust
# teacher_model=cartl_wrn34_cifar100_4_0.01-best_robust
# teacher_model=wd_fdm_True_res18_svhn_5_1.0-best_robust
# teacher_model=at_res18_svhn-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_4_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_6_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_8_1.0-best_robust
# teacher_model=new_wd_fdm_False_wrn28-4_cifar10_8_1.0-last
# teacher_model=cartl_wrn34_cifar100_8_0.01-best_robust
# teacher_model=cartl_wrn34_cifar100_6_0.01-best_robust
# teacher_model=at_wrn34_cifar100-best_robust

dataset=gtsrb
num_classes=43
model_arch=pwrn28\(4\)
total_size=12630

for k in 6; do
    echo "#######################################################"
    echo "neft transferring from ${teacher_model}"

    echo "freeze bn layer"
    python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
    -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4 --freeze-bn  --reuse-teacher-statistic
    valid $?

    # echo "pgd20 attack"

    # freeze_bn_model_path=trained_models/sntl_1_0.4_rts_True_${model_arch}_${dataset}_${k}_${teacher_model}-last

    # python -m exps.eval_robust_pwrn34 --model=${freeze_bn_model_path} -k=${k} \
    #         --log=rts_new_neft_${model_arch}_${dataset}_pgd20.log --result-file=logs/rts_new_neft_${model_arch}_${dataset}_pgd20.json \
    #         --model-type=${model_arch} --dataset=${dataset} --num_classes=${num_classes}
    # valid $?


    trained_model_path=trained_models/sntl_1_0.4_rts_True_${model_arch}_${dataset}_${k}_${teacher_model}-last
    echo ${trained_model_path}
    echo "pgd100 attack"
    
    python -m exps.foolbox_bench \
            --model=${trained_model_path} \
            -k=${k} \
            --log=tl_${model_arch}_${total_size}_${dataset}_pgd100.log \
            --result-file=logs/neft_${model_arch}_${total_size}_${dataset}_pgd100.json \
            --model-type=${model_arch} \
            --dataset=${dataset} --num_classes=${num_classes} --total-size ${total_size}
    valid $?
done
