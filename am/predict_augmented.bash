#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J predict_augmented
#SBATCH --exclude=sls-sm-[5]

echo "STARTING AUGMENTED ACOUSTIC MODEL PREDICTION JOB"

. ./path.sh
. $MENG_ROOT/am/augmented_config.sh
. $MENG_ROOT/am/path-opt.sh

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
    log_dir=$LOGS/$EXPT_NAME/adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/adversarial_fc_${ADV_FC_DELIM}_act_${ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    log_dir=$LOGS/$EXPT_NAME/gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/gan_fc_${GAN_FC_DELIM}_act_${GAN_FC_DELIM}_${run_mode}_ratio${NOISE_RATIO}
else
    log_dir=$LOGS/$EXPT_NAME/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/${run_mode}_ratio${NOISE_RATIO}
fi

mkdir -p $log_dir/predict_${PREDICT_DOMAIN}
predict_log=$log_dir/predict_${PREDICT_DOMAIN}/predictions.log

echo "Source domain ${SOURCE_DOMAIN}, target domain ${TARGET_DOMAIN}, predict domain ${PREDICT_DOMAIN}"
echo "Logging results to $predict_log"

echo "Predicting using TDNN..."
# Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $DATASET/${PREDICT_DOMAIN}-dev-norm.blogmel.scp \
    --param $model_dir/param-$MODEL_EPOCH \
    --label $DATASET/ihm-pdfids.txt \
    > $predict_log
echo "Done predicting using TDNN."

# Get FER for run
python $MENG_ROOT/am/eval-frames.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali

# Evaluate errors
python $MENG_ROOT/am/err_analysis.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali $log_dir/predict_${PREDICT_DOMAIN}

echo "DONE AUGMENTED ACOUSTIC MODEL PREDICTION JOB"
