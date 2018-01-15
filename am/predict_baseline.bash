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
. $MENG_ROOT/am/job_config.sh
. $MENG_ROOT/am/path-opt.sh

mkdir -p $LOGS/$EXPT_NAME
predict_log=$LOGS/$EXPT_NAME/predict_baseline.log

echo "Predicting using TDNN..."
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $DATASET/${DOMAIN}-dev-norm.blogmel.scp \
    --param $MODEL_DIR/param-$MODEL_EPOCH \
    --label $DATASET/${DOMAIN}-pdfids.txt \
    > $predict_log
echo "Done predicting using TDNN."

echo "DONE BASELINE ACOUSTIC MODEL PREDICTION JOB"
