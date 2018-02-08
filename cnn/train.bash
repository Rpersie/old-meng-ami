#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_cnn_md

echo "STARTING CONVOLUTIONAL MULTIDECODER TRAINING JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $TRAIN_ENV
echo "Environment set up."

if [ "$#" -le 1 ]; then
    echo "Run mode not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

adversarial=false
if [ "$#" -ge 2 ]; then
    if [ "$2" == "adversarial" ]; then
        adversarial=true
        echo "Using adversarial training"
    fi
fi

mkdir -p $LOGS/$EXPT_NAME
if [ "$adversarial" == true ]; then
    train_log=$LOGS/$EXPT_NAME/train_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}.log
    if [ -f $train_log ]; then
        # Move old log
        mv $train_log $LOGS/$EXPT_NAME/train_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}-$(date +"%F_%T%z").log
    fi
else
    train_log=$LOGS/$EXPT_NAME/train_${run_mode}.log
    if [ -f $train_log ]; then
        # Move old log
        mv $train_log $LOGS/$EXPT_NAME/train_${run_mode}-$(date +"%F_%T%z").log
    fi
fi

if [ "$PROFILE_RUN" = true ] ; then
    echo "Profiling..."
    python3 cnn/scripts/train_md.py ${run_mode} ${adversarial} profile > $train_log
    echo "Profiling done"
    # echo "Profiling done -- please run 'snakeviz --port=8890 --server $LOGS/$EXPT_NAME/train_${run_mode}.prof' to view the results in browser"
else
    python3 cnn/scripts/train_md.py ${run_mode} ${adversarial} > $train_log
fi

echo "DONE CONVOLUTIONAL MULTIDECODER TRAINING JOB"
