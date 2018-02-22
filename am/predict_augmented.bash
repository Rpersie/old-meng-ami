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

if [ "$#" -lt 4 ]; then
    echo "Run mode, source domain, target domain and predict domain not specified; exiting"
    exit 1
fi

run_mode=$1
echo "Using run mode ${run_mode}"

src_domain=$2
tar_domain=$3
pre_domain=$4
echo "Source domain ${src_domain}, target domain ${tar_domain}, predict domain ${pre_domain}"

domain_adversarial=false
gan=false
if [ "$#" -ge 4 ]; then
    if [ "$5" == "domain" ]; then
        domain_adversarial=true
        echo "Using domain_adversarial training"
    fi
    
    if [ "$5" == "gan" ]; then
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
    model_dir=$MODEL_DIR/$expt_name/domain_adversarial_fc_${DOMAIN_ADV_FC_DELIM}_act_${DOMAIN_ADV_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
elif [ "$gan" == true ]; then
    log_dir=$LOGS/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/$expt_name/gan_fc_${GAN_FC_DELIM}_act_${GAN_ACTIVATION}_${run_mode}_ratio${NOISE_RATIO}
else
    log_dir=$LOGS/$expt_name/${run_mode}_ratio${NOISE_RATIO}
    mkdir -p $log_dir
    model_dir=$MODEL_DIR/$expt_name/${run_mode}_ratio${NOISE_RATIO}
fi

mkdir -p $log_dir/predict_${pre_domain}
predict_log=$log_dir/predict_${pre_domain}/predictions.log

echo "Logging results to $predict_log"

echo "Predicting using TDNN..."
# Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $DATASET/${pre_domain}-dev-norm.blogmel.scp \
    --param $model_dir/param-$MODEL_EPOCH \
    --label $DATASET/ihm-pdfids.txt \
    > $predict_log
echo "Done predicting using TDNN."

# Get FER for run
python $MENG_ROOT/am/eval-frames.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali

# Evaluate errors
python $MENG_ROOT/am/err_analysis.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali $log_dir/predict_${pre_domain}

echo "DONE AUGMENTED ACOUSTIC MODEL PREDICTION JOB"
