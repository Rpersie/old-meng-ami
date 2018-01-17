#!/bin/bash
#SBATCH -p sm
#SBATCH -n1
#SBATCH -N1-1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=48:00:00
#SBATCH -J predict_augmented

echo "STARTING AUGMENTED ACOUSTIC MODEL PREDICTION JOB"

. ./path.sh
. $MENG_ROOT/am/augmented_config.sh
. $MENG_ROOT/am/path-opt.sh

mkdir -p $LOGS/$EXPT_NAME/predict_${PREDICT_DOMAIN}
predict_log=$LOGS/$EXPT_NAME/predict_${PREDICT_DOMAIN}/predictions.log

echo "Source domain ${SOURCE_DOMAIN}, target domain ${TARGET_DOMAIN}, predict domain ${PREDICT_DOMAIN}"

echo "Predicting using TDNN..."
OPENBLAS_CORETYPE=Sandybridge OMP_NUM_THREADS=4 /data/sls/scratch/haotang/ami/dist/nn-20171210-4c6c341-openblas/nnbin/frame-tdnn-predict \
    --frame-scp $DATASET/${PREDICT_DOMAIN}-dev-norm.blogmel.scp \
    --param $MODEL_DIR/param-$MODEL_EPOCH \
    --label $DATASET/${PREDICT_DOMAIN}-pdfids.txt \
    > $predict_log
echo "Done predicting using TDNN."

# Get FER for run
python $MENG_ROOT/am/eval-frames.py $predict_log $GOLD_DIR/${TARGET_DOMAIN}-dev-tri3.bali

echo "DONE AUGMENTED ACOUSTIC MODEL PREDICTION JOB"
