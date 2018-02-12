#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J predict_baseline

echo "STARTING BASELINE ACOUSTIC MODEL PREDICTION JOB"

. ./path.sh
. $MENG_ROOT/am/baseline_config.sh
. $MENG_ROOT/am/path-opt.sh

mkdir -p $LOGS/$EXPT_NAME/predict_${PREDICT_DOMAIN}
predict_log=$LOGS/$EXPT_NAME/predict_${PREDICT_DOMAIN}/predictions.log

echo "Train domain ${TRAIN_DOMAIN}, predict domain ${PREDICT_DOMAIN}"

echo "Predicting using TDNN..."
# Always use IHM pdfids, even for SDM1 (data are parallel -- see Hao email from 1/17/18)
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $DATASET/${PREDICT_DOMAIN}-dev-norm.blogmel.scp \
    --param $MODEL_DIR/param-$MODEL_EPOCH \
    --label $DATASET/ihm-pdfids.txt \
    > $predict_log
echo "Done predicting using TDNN."

# Get FER for run
python $MENG_ROOT/am/eval-frames.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali

# Evaluate errors
python $MENG_ROOT/am/err_analysis.py $predict_log $GOLD_DIR/ihm-dev-tri3.bali $LOGS/$EXPT_NAME/predict_${PREDICT_DOMAIN}

echo "DONE BASELINE ACOUSTIC MODEL PREDICTION JOB"
