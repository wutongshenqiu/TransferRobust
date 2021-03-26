#!/bin/bash
source scripts/utils.sh

cuda $1

# teacher_model=wd_fdm_True_wrn28\(4\)_cifar10_8_1.0-best_robust
# teacher_model=cartl_wrn28\(4\)_cifar10_6_0.01-best_robust
# teacher_model=at_wrn28\(4\)_cifar10-best_robust
# teacher_model=cartl_wrn28\(4\)_gtsrb_6_0.01-best_robust
# teacher_model=at_wrn28\(4\)_gtsrb-best_robust
teacher_model=cartl_wrn34_cifar100_7_0.005-best_robust
# teacher_model=wd_fdm_True_res18_svhn_5_1.0-best_robust
# teacher_model=at_res18_svhn-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_4_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_6_1.0-best_robust
# teacher_model=wd_fdm_True_wrn34_cifar100_8_1.0-best_robust

dataset=cifar10
num_classes=10
model_arch=pwrn34
total_size=60000
k=7
seed=751

echo "#######################################################"
echo "neft transferring from ${teacher_model}"

# echo "bn + rts"
# python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
#        -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4 --freeze-bn --reuse-teacher-statistic
# valid $?

# echo "rts"
# python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
#        -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4 --reuse-teacher-statistic

echo "bn"
python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
       -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4 --freeze-bn

# python -m exps.neft_spectrum_norm -m ${model_arch} -n ${num_classes} \
#        -d ${dataset} -k ${k} -t ${teacher_model} --norm-beta 0.4


# trained_model_path1=trained_models/sntl_1_0.4_rts_True_${model_arch}_${dataset}_${k}_${teacher_model}-last
# trained_model_path2=trained_models/sntl_1_0.4_rts_False_${model_arch}_${dataset}_${k}_${teacher_model}-last
trained_model_path3=trained_models/sntl_1_0.4_True_${model_arch}_${dataset}_${k}_${teacher_model}_${seed}-last
# trained_model_path4=trained_models/sntl_1_0.4_False_${model_arch}_${dataset}_${k}_${teacher_model}-last

for trained_model_path in $trained_model_path3; do
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
                --log=tl_${model_arch}_${total_size}_${dataset}_pgd100.log \
                --result-file=logs/neft_${model_arch}_${total_size}_${dataset}_pgd100.json \
                --model-type=${model_arch} \
                --dataset=${dataset} --num_classes=${num_classes} --total-size ${total_size}
        valid $?
done
