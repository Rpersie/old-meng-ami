#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_ae_cnn_md

echo "STARTING CONVOLUTIONAL MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
train_log=$LOGS/$EXPT_NAME/train_ae.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOGS/$EXPT_NAME/train_ae-$(date +"%F_%T%z").log
fi

if [ "$PROFILE_RUN" = true ] ; then
    echo "Profiling..."
    python3 cnn/scripts/train_md.py ae profile > $train_log
    echo "Profiling done -- please run 'snakeviz --port=8890 --server $LOGS/$EXPT_NAME/train_ae.prof' to view the results in browser"
else
    python3 cnn/scripts/train_md.py ae > $train_log
fi

echo "DONE CONVOLUTIONAL MULTIDECODER TRAINING JOB"
