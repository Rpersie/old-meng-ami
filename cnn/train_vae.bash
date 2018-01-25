#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_vae_cnn_md
#SBATCH --exclude=sls-sm-[5,6,7]

echo "STARTING CONVOLUTIONAL VARIATIONAL MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
train_log=$LOGS/$EXPT_NAME/train_vae.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOGS/$EXPT_NAME/train_vae-$(date +"%F_%T%z").log
fi

./cnn/train_vae.sh > $train_log

echo "DONE CONVOLUTIONAL VARIATIONAL MULTIDECODER TRAINING JOB"
