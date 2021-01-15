
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

TEACHER_MODEL=at_wrn34_cifar100-best_robust


for r in 0.5 0.2 0.1; do

    echo "#######################################################"
    echo "lwf transfer from ${TEACHER_MODEL}"

    cli lwf -m wrn34 -n 10 -d cifar10\(${r}\) -l 0.1 -t ${TEACHER_MODEL}
    valid
done
