
###############################################
# check CUDA_VISIBLE_DEVICES is set or not
if [[ -z ${CUDA_VISIBLE_DEVICES} ]]; then
    if [ $1 ]; then
        # if not set, try read args
        export CUDA_VISIBLE_DEVICES=$1
    else
        # else manually set CUDA_VISIBLE_DEVICES being 0
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
echo "CUDA:${CUDA_VISIBLE_DEVICES} is available"


function valid () {
    # check whether previous cmd is correctly executed
    if [ $? -ne 0 ]; then
        exit
    fi
}
###############################################

# TEACHER_MODEL=at_res18_svhn-best_robust
# TEACHER_MODEL=at_wrn28\(4\)_gtsrb-best_robust
TEACHER_MODEL=at_wrn34_cifar100-best_robust
dataset=cifar10
model_arch=wrn34

lam=0.005
num_classes=10

echo "#######################################################"
echo "lwf transfer from ${TEACHER_MODEL}"

cli lwf -m ${model_arch} -n ${num_classes} -d ${dataset} -l ${lam} -t ${TEACHER_MODEL}
valid $?

# python -m exps.eval_robust_wrn34 --model=trained_models/lwf_${model_arch}_${dataset}_${lam}-best \
#         --log=lwf.log --result-file=logs/lwf.json --model-type=${model_arch} --dataset=${dataset} --num_classes=${num_classes}
# valid $?

python -m exps.foolbox_bench --model=trained_models/lwf_${model_arch}_${dataset}_${lam}_1024-best \
        --log=lwf_pgd100_attack.log --result-file=logs/final_lwd_pgd100_attack.json --model-type=${model_arch} \
        --dataset=${dataset} --num_classes=${num_classes} --total-size 60000
valid $?

