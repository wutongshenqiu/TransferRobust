
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

# MODEL_PATH=trained_models

model_path = {
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.001_4_from_cartl_wrn34_cifar100_4_0.01-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.0006_4_from_cartl_wrn34_cifar100_4_0.01-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.0003_4_from_cartl_wrn34_cifar100_4_0.01-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.001_4_from_cartl_wrn34_cifar100_4_0.005-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.0006_4_from_cartl_wrn34_cifar100_4_0.005-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.0003_4_from_cartl_wrn34_cifar100_4_0.005-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.001_8_from_cartl_wrn34_cifar100_4_0.01-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.0006_8_from_cartl_wrn34_cifar100_4_0.01-best_robust.log",
    "logs/bn_freeze_ptl_pwrn34_cifar10_0.0003_8_from_cartl_wrn34_cifar100_4_0.01-best_robust.log",
}

for b in 0.001 0.0006 0.0003; do
    TEACHER_MODEL=cartl_wrn34_cifar100_${k}_${l}-best_robust

    echo "#######################################################"
    echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"

    screen cli ptl -m pwrn34 -n 10 -d cifar10 -b ${b} -k ${k} -t ${TEACHER_MODEL}
    valid
done
