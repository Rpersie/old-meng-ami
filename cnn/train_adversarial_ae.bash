#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_adversarial_ae_cnn_md

echo "STARTING CONVOLUTIONAL ADVERSARIAL MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

mkdir -p $LOGS/$EXPT_NAME
train_log=$LOGS/$EXPT_NAME/train_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_ae.log
if [ -f $train_log ]; then
    # Move old log
    mv $train_log $LOGS/$EXPT_NAME/train_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_ae-$(date +"%F_%T%z").log
fi

if [ "$PROFILE_RUN" = true ] ; then
    echo "Profiling..."
    python3 cnn/scripts/train_adversarial_md.py ae profile > $train_log
    echo "Profiling done -- please run 'snakeviz --port=8890 --server $LOGS/$EXPT_NAME/train_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_ae.prof' to view the results in browser"
else
    python3 cnn/scripts/train_adversarial_md.py ae > $train_log
fi

echo "DONE CONVOLUTIONAL ADVERSARIAL MULTIDECODER TRAINING JOB"
