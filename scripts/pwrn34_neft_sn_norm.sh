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

MODEL_PATH=trained_models
POWER_ITER=1
NORM_BETA=0.6

for k in 4 6 8; do
    TEACHER_MODEL=cartl_wrn34_cifar100_${k}_0.01-best_robust

    echo "#######################################################"
    echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"
    python -m exps.neft_spectrum_norm --model pwrn34 --num_classes 10 --dataset cifar10 --teacher ${TEACHER_MODEL} -k ${k} --power-iter ${POWER_ITER} --norm-beta ${NORM_BETA}
    valid

    python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_pwrn34_cifar10_${k}_${TEACHER_MODEL}-last -k=${k} \
            --log=sntl.log --result-file=logs/sntl.json
    valid

done

TEACHER_MODEL=cartl_wrn34_cifar100_8_1.0-best_robust
echo "#######################################################"
echo "using NEFT(SN) to transfer from ${TEACHER_MODEL}"
python -m exps.neft_spectrum_norm --model pwrn34 --num_classes 10 --dataset cifar10 --teacher ${TEACHER_MODEL} -k 8  --power-iter ${POWER_ITER} --norm-beta ${NORM_BETA}
valid
python -m exps.eval_robust_pwrn34 --model=${MODEL_PATH}/sntl_${POWER_ITER}_${NORM_BETA}_pwrn34_cifar10_8_${TEACHER_MODEL}-last -k=8 \
        --log=sntl.log --result-file=logs/sntl.json
valid

