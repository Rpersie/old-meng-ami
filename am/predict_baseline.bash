#!/bin/bash
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J predict_baseline
#SBATCH --exclude=sls-sm-[5],sls-tesla-[0,1]

echo "STARTING BASELINE ACOUSTIC MODEL PREDICTION JOB"

. ./path.sh
. $MENG_ROOT/am/baseline_config.sh
. $MENG_ROOT/am/path-opt.sh

if [ "$#" -lt 2 ]; then
    echo "Train domain and predict domain not specified; exiting"
    exit 1
fi

train_domain=$1
predict_domain=$2
echo "Train domain ${train_domain}, predict domain ${predict_domain}"

expt_name="train_${train_domain}/baseline/${ARCH_NAME}"

mkdir -p $LOG_DIR/$expt_name/predict_${predict_domain}
predict_log=$LOG_DIR/$expt_name/predict_${predict_domain}/predictions.log

model_dir=$MODEL_DIR/$expt_name

echo "Predicting using TDNN..."
# Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $DATASET/${predict_domain}-dev-norm.blogmel.scp \
    --param $model_dir/param-$MODEL_EPOCH \
    --label $DATASET/ihm-pdfids.txt \
    > $predict_log
echo "Done predicting using TDNN."

# Get FER for run
python $MENG_ROOT/am/eval-frames.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali

# Evaluate errors
python $MENG_ROOT/am/err_analysis.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali $LOG_DIR/$expt_name/predict_${predict_domain}

echo "DONE BASELINE ACOUSTIC MODEL PREDICTION JOB"
