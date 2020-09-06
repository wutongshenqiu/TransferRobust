#!/bin/bash
DEVICE=cuda
# check input parameters for the script,
# assuming the first one is specifying device.
if [ $# -ge 1 ]; then 
    DEVICE=$1
fi

# assuming the second one is config path
CONFIG_PATH=src/config.env
if [ $# -ge 2 ]; then
    CONFIG_PATH=$2
fi

if [ ! -x ${CONFIG_PATH} ]; then
    PWD=`pwd`
    echo  "Current path is: ${PWD}. I cannot find: ${CONFIG_PATH}"
    exit
fi

# replace device configuration
echo "DEVICE=${DEVICE} >> ${CONFIG_PATH}"
sed -i -e "s/DEVICE=.*$/DEVICE=${DEVICE}/g" ${CONFIG_PATH}
