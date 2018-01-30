#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_dae_cnn_md
#SBATCH --exclude=sls-sm-[5,6,7]

echo "STARTING CONVOLUTIONAL DENOISING MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
train_log=$LOGS/$EXPT_NAME/train_dae_ratio${NOISE_RATIO}.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOGS/$EXPT_NAME/train_dae_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
fi

python3 cnn/scripts/train.py dae > $train_log

echo "DONE CONVOLUTIONAL DENOISING MULTIDECODER TRAINING JOB"
