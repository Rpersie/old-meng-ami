#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J augment_cnn_md
#SBATCH --exclude=sls-sm-[5]

echo "STARTING CONVOLUTIONAL MULTIDECODER DATA AUGMENTATION JOB"

. ./path.sh
. ./cnn/job_config.sh

echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $AUGMENT_ENV
echo "Environment set up."

if [ "$#" -lt 1 ]; then
    echo "Run mode not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

adversarial=false
gan=false
if [ "$#" -ge 2 ]; then
    if [ "$2" == "adversarial" ]; then
        adversarial=true
        echo "Using adversarial training"
    fi
    
    if [ "$2" == "gan" ]; then
        gan=true
        echo "Using generative adversarial net (GAN) style training"
    fi
fi

mkdir -p $LOGS/$EXPT_NAME
if [ "$adversarial" == true ]; then
    augment_log=$LOGS/$EXPT_NAME/augment_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}.log
    if [ -f $augment_log ]; then
        # Move old log
        mv $augment_log $LOGS/$EXPT_NAME/augment_adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}-$(date +"%F_%T%z").log
    fi
    mkdir -p $AUGMENTED_DATA_DIR/adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    augment_log=$LOGS/$EXPT_NAME/augment_gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $augment_log ]; then
        # Move old log
        mv $augment_log $LOGS/$EXPT_NAME/augment_gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
    mkdir -p $AUGMENTED_DATA_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
else
    augment_log=$LOGS/$EXPT_NAME/augment_${run_mode}_ratio${NOISE_RATIO}.log
    if [ -f $augment_log ]; then
        # Move old log
        mv $augment_log $LOGS/$EXPT_NAME/augment_${run_mode}_ratio${NOISE_RATIO}-$(date +"%F_%T%z").log
    fi
    mkdir -p $AUGMENTED_DATA_DIR/${run_mode}_ratio${NOISE_RATIO}
fi

python3 cnn/scripts/augment_md.py ${run_mode} ${adversarial} ${gan} > $augment_log

echo "DONE CONVOLUTIONAL MULTIDECODER DATA AUGMENTATION JOB"
