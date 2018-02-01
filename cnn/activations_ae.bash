#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J activations_ae_cnn_md
#SBATCH --exclude=sls-sm-[5]

echo "STARTING CONVOLUTIONAL MULTIDECODER ACTIVATION LOGGING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $ACTIVATIONS_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
activations_log=$LOGS/$EXPT_NAME/activations_top${TOP_COUNT}_ae.log
if [ -f $activations_log ]; then
    # Move old log
    mv $activations_log $LOGS/$EXPT_NAME/activations_top${TOP_COUNT}_ae-$(date +"%F_%T%z").log
fi

mkdir -p $ACTIVATIONS_DIR/ae_ratio${NOISE_RATIO}

python3 cnn/scripts/activations.py ae > $activations_log

echo "DONE CONVOLUTIONAL MULTIDECODER ACTIVATION LOGGING JOB"
