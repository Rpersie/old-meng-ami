#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J train_augmented
#SBATCH --exclude=sls-sm-[5],sls-tesla-[0,1]

echo "STARTING AUGMENTED ACOUSTIC MODEL TRAINING JOB"

. ./path.sh
. $MENG_ROOT/am/augmented_config.sh
. $MENG_ROOT/am/path-cuda.sh

if [ "$#" -lt 3 ]; then
    echo "Run mode, source domain and target domain not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

src_domain=$2
tar_domain=$3
echo "Source domain ${src_domain}, target domain ${tar_domain}"

domain_adversarial=false
gan=false
if [ "$#" -ge 3 ]; then
    if [ "$4" == "domain" ]; then
        domain_adversarial=true
        echo "Using domain_adversarial training"
    fi
    
    if [ "$4" == "gan" ]; then
        gan=true
        echo "Using generative adversarial net (GAN) style training"
    fi
fi

expt_name="train_${tar_domain}/augmented_src_${src_domain}/${ARCH_NAME}/${CNN_NAME}"

mkdir -p $LOGS/$expt_name
mkdir -p $MODEL_DIR/$expt_name

if [ "$domain_adversarial" == true ]; then
    log_dir=$LOGS/$expt_name/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/$expt_name/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
elif [ "$gan" == true ]; then
    log_dir=$LOGS/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
else
    log_dir=$LOGS/$expt_name/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    augmented_data_dir=$AUGMENTED_DATA_BASE_DIR/${run_mode}_ratio${NOISE_RATIO}
    model_dir=$MODEL_DIR/$expt_name/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $model_dir
fi

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
        --frame-scp $augmented_data_dir/train-src_${src_domain}-tar_${tar_domain}.scp \
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
