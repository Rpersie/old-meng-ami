#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J augment_dvae_cnn_md
#SBATCH --exclude=sls-sm-[1,2,4]

echo "STARTING CONVOLUTIONAL DENOISING VARIATIONAL MULTIDECODER DATA AUGMENTATION JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
augment_log=$LOGS/$EXPT_NAME/augment_dvae_ratio${NOISE_RATIO}.log
if [ -f $augment_log ]; then
    # Move old log
    mv $augment_log $LOGS/$EXPT_NAME/augment_dvae_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
fi

mkdir -p $AUGMENTED_DATA_DIR/dvae_ratio${NOISE_RATIO}

python3 cnn/scripts/augment.py dvae > $augment_log

echo "DONE CONVOLUTIONAL DENOISING VARIATIONAL MULTIDECODER DATA AUGMENTATION JOB"
