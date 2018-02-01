#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J augment_vae_cnn_md
#SBATCH --exclude=sls-sm-[1,2,4]

echo "STARTING CONVOLUTIONAL VARIATIONAL MULTIDECODER DATA AUGMENTATION JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
augment_log=$LOGS/$EXPT_NAME/augment_vae.log
if [ -f $augment_log ]; then
    # Move old log
    mv $augment_log $LOGS/$EXPT_NAME/augment_vae-$(date +"%F_%T%z").log
fi

mkdir -p $AUGMENTED_DATA_DIR/vae_ratio${NOISE_RATIO}

python3 cnn/scripts/augment.py vae > $augment_log

echo "DONE CONVOLUTIONAL VARIATIONAL MULTIDECODER DATA AUGMENTATION JOB"
