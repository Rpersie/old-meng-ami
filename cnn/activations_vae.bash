#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J activations_vae_cnn_md
#SBATCH --exclude=sls-sm-[5]

echo "STARTING CONVOLUTIONAL MULTIDECODER ACTIVATION VISUALIZATION JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $ACTIVATIONS_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
activations_log=$LOGS/$EXPT_NAME/activations_top${TOP_COUNT}_vae.log
if [ -f $activations_log ]; then
    # Move old log
    mv $activations_log $LOGS/$EXPT_NAME/activations_top${TOP_COUNT}_vae-$(date +"%F_%T%z").log
fi

mkdir -p $ACTIVATIONS_DIR/vae

python3 cnn/scripts/activations.py vae > $activations_log

echo "DONE CONVOLUTIONAL MULTIDECODER ACTIVATION VISUALIZATION JOB"
