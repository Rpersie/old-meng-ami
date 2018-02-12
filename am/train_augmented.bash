#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_augmented
#SBATCH --exclude=sls-sm-[5]

echo "STARTING AUGMENTED ACOUSTIC MODEL TRAINING JOB"

. ./path.sh
. $MENG_ROOT/am/augmented_config.sh
. $MENG_ROOT/am/path-cuda.sh

if [ "$#" -le 1 ]; then
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
    log_dir=$LOGS/$EXPT_NAME/adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${MODEL_TYPE}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${MODEL_TYPE}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
elif [ "$gan" == true ]; then
    log_dir=$LOGS/$EXPT_NAME/gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${MODEL_TYPE}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${MODEL_TYPE}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
else
    log_dir=$LOGS/$EXPT_NAME/${run_mode}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/${MODEL_TYPE}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/${MODEL_TYPE}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
fi

echo "Source domain ${SOURCE_DOMAIN}, target domain ${TARGET_DOMAIN}"
echo "Using log directory $log_dir"

for epoch in $(seq $START_EPOCH $END_EPOCH); do
    echo "========== EPOCH $epoch =========="

    epoch_log=$log_dir/train_augmented-epoch${epoch}.log

    if [ ! -f $model_dir/param-$((epoch-1)) ]; then
        # Parameter file doesn't exist -- only generate if we're just starting
        if [ "$epoch" -eq "1" ]; then
            echo "TDNN not initialized. Initializing parameters..."
            $MENG_ROOT/am/init-tdnn.py random > $model_dir/param-$((epoch-1))
            echo "Done initializing parameters."
        else
            echo "Parameter file does not exist for (epoch - 1 = $((epoch-1)))"
            exit 1
        fi
    fi

    # Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
    OMP_NUM_THREADS=1 /data/sls/scratch/haotang/ami/dist/nn-20171213-4c6c341/nnbin/frame-tdnn-learn-gpu \
        --frame-scp $augmented_data_dir/train-src_${SOURCE_DOMAIN}-tar_${TARGET_DOMAIN}.scp \
        --label-scp $DATASET/ihm-train-tri3.bali.scp \
        --param $model_dir/param-$((epoch-1)) \
        --opt-data $model_dir/opt-data-$((epoch-1)) \
        --output-param $model_dir/param-$epoch \
        --output-opt-data $model_dir/opt-data-$epoch \
        --label $DATASET/ihm-pdfids.txt \
        --seed $epoch \
        --shuffle \
        --opt const-step \
        --step-size 0.05 \
        --clip 5 \
        > $epoch_log
done

echo "DONE AUGMENTED ACOUSTIC MODEL TRAINING JOB"
