#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J activations_cnn_md

echo "STARTING CONVOLUTIONAL MULTIDECODER ACTIVATION LOGGING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $ACTIVATIONS_ENV
echo "Environment set up."

if [ "$#" -ne 1 ]; then
    echo "Run mode not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

mkdir -p $LOGS/$EXPT_NAME
activations_log=$LOGS/$EXPT_NAME/activations_top${TOP_COUNT}_${run_mode}.log
if [ -f $activations_log ]; then
    # Move old log
    mv $activations_log $LOGS/$EXPT_NAME/activations_top${TOP_COUNT}_${run_mode}-$(date +"%F_%T%z").log
fi

mkdir -p $ACTIVATIONS_DIR/${run_mode}_ratio${NOISE_RATIO}

python3 cnn/scripts/activations_md.py ${run_mode} > $activations_log

echo "DONE CONVOLUTIONAL MULTIDECODER ACTIVATION LOGGING JOB"
