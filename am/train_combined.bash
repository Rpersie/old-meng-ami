#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_combined
#SBATCH --exclude=sls-sm-[5],sls-tesla-[0,1]

echo "STARTING BASELINE + AUGMENTED ACOUSTIC MODEL TRAINING JOB"

. ./path.sh
. $MENG_ROOT/am/combined_config.sh
. $MENG_ROOT/am/path-cuda.sh

if [ "$#" -lt 1 ]; then
    echo "Run mode not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

domain_adversarial=false
gan=false
if [ "$#" -ge 2 ]; then
    if [ "$2" == "domain" ]; then
        domain_adversarial=true
        echo "Using domain_adversarial training"
    fi

    if [ "$2" == "gan" ]; then
        gan=true
        echo "Using generative adversarial net (GAN) style training"
    fi
fi

mkdir -p $LOG_DIR/$EXPT_NAME
if [ "$domain_adversarial" == true ]; then
    log_dir=$LOG_DIR/$EXPT_NAME/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
elif [ "$gan" == true ]; then
    log_dir=$LOG_DIR/$EXPT_NAME/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
else
    log_dir=$LOG_DIR/$EXPT_NAME/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/${run_mode}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
fi

# Create combined SCP file for IHM baseline + SDM1 augmented
cat $DATASET/ihm-train-norm.blogmel.scp $augmented_data_dir/train-src_ihm-tar_sdm1.scp > $augmented_data_dir/train-combined.blogmel.scp
sed -e 's/^/src_ihm_tar_sdm1_/' $DATASET/ihm-train-tri3.bali.scp > $augmented_data_dir/src_ihm_tar_sdm1-train-tri3.bali.scp
cat $DATASET/ihm-train-tri3.bali.scp $augmented_data_dir/src_ihm_tar_sdm1-train-tri3.bali.scp > $augmented_data_dir/combined-train-tri3.bali.scp

for epoch in $(seq $START_EPOCH $END_EPOCH); do
    echo "========== EPOCH $epoch =========="

    step_size=$(echo "scale=10; 0.05 * 0.75 ^ ($epoch - 1)" | bc)
    echo "Step size now $step_size"

    epoch_log=$log_dir/train_combined-epoch${epoch}.log

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
        --frame-scp $augmented_data_dir/train-combined.blogmel.scp \
        --label-scp $augmented_data_dir/combined-train-tri3.bali.scp \
        --param $model_dir/param-$((epoch-1)) \
        --opt-data $model_dir/opt-data-$((epoch-1)) \
        --output-param $model_dir/param-$epoch \
        --output-opt-data $model_dir/opt-data-$epoch \
        --label $DATASET/ihm-pdfids.txt \
        --seed $epoch \
        --shuffle \
        --opt const-step \
        --step-size $step_size \
        --clip 5 \
        > $epoch_log
done

echo "DONE BASELINE + AUGMENTED ACOUSTIC MODEL TRAINING JOB"
